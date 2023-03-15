from typing import List, Set
import numpy.random as random
from collections import defaultdict
from itertools import combinations
from ..helpers.utils import choice_np, sorted_item_set
from ..helpers.event import Event
from ..helpers.sst.proposition import ProbProposition 
from ..helpers.event_calc import DECStory, from_world_event, from_world_q_event
from ..Entities import Entity

def prop_factory(a: Entity, b: Entity, belief: float = 1.0, try_reverse: bool = False) -> List[ProbProposition]:
    """
    Return list of propositions that hold between two entities based on their types.
    

    Parameters
    ----------
    a : Entity
        Source entity.
    b : Entity
        Object entity.
    belief : float, optional
        Belief probability for the proposition. The default is 1.0.

    Returns
    -------
    List[ProbProposition]
        List of propositions that hold for entities a, b.

    """
    props = []
    if a.kind == "object" and b.kind == "person":
        props += [ProbProposition(name="held", args=(a.name, b.name), belief=belief)]
    elif b.kind == "location" and a.kind in ["object", "person"]:
        props += [ProbProposition(name="at", args=(a.name, b.name), belief=belief)]
    elif b.kind == "indef_location" and a.kind == "person":
        # result of indef action
        props += [ProbProposition(name="at", args=(a.name, b.location1.name), belief=0.5),
                  ProbProposition(name="at", args=(a.name, b.location2.name), belief=0.5)]
    if try_reverse:
        props += prop_factory(b, a, belief, try_reverse=False)
    
    return props
    
def create_prop_maps(entities):
        """ 
        Create maps of proposition to idx and vice versa for all entities in the world.
        """
        prop2idx = {}
        idx2prop = {}
        props = []
        for a, b in combinations(entities, 2):
            props += [p.realization for p in prop_factory(a, b, try_reverse=True)]
        
        # ensure deterministic given same entity set
        for i, p in enumerate(sorted(list(set(props)))):
            prop2idx[p] = i
            idx2prop[i] = p
        
        return idx2prop, prop2idx



class World(object):
    """
    The world in which a bAbI game takes place. The game is the engine that allows us to create bAbI stories that have
    coherent rules. The desired dataset we wish to create simply describes bAbI games.
    The world object contains all entities, actions, questions and rules that governs a bAbI game in this world.
    """
    def __init__(self, entities=None, action_list=None, question_list=None, params=None):
        self.entities = entities
        self.action_list = action_list
        self.question_list = question_list
        self.params = params
        self.timestep = 0
        self.history = {}
        self.q_history = {}
        self.ent_map = {}
        self.locations = []
        self.known_items_history = {}
        self.diff_props = defaultdict(list)
        self.current_seed = -1
        self.idx2prop = None
        self.prop2idx = None

    def populate(self, entities):
        self.entities = entities
        self.allocate()
        self.idx2prop, self.prop2idx = create_prop_maps(self.entities)
        self.ent_map = {e.name: e.kind for e in entities}
        

    def rule(self, params, action_list, question_list):
        """
        Sets the rules of the world:
        :param params: will determine which actions and questions will be generated and with what probability
        :param action_list: will manage the list of possible actions
        :param question_list: will manage the list of possible questions
        """
        self.params = params
        self.action_list = action_list
        self.question_list = question_list

    def allocate(self):
        """
        Assigns each non-location entity in the world a (starting) location
        """
        locations = []
        for entity in sorted_item_set(self.entities):
            if entity.kind == "location":
                locations.append(entity)
        for entity in sorted_item_set(self.entities):
            if entity.kind == "person" or entity.kind == "object":
                location = choice_np(locations)
                entity.holder = location
                location.holds.add(entity)
        self.locations = locations
    
    def mentioned_people(self):
        """ 
        Return list of people mentioned thus far in story
        """
        mentioned = []
        for k, v in self.history.items():
            curr_ments = []
            if type(v.source) == list:
                curr_ments += [self.get_entity_by_name(t)for t in v.source]
            else:
                curr_ments += [self.get_entity_by_name(v.source)]
                
            if type(v.target) == list:
                curr_ments += [self.get_entity_by_name(t)for t in v.target]
            else:
                curr_ments += [self.get_entity_by_name(v.target)]
                
            curr_people = [m for m in curr_ments if self.ent_map.get(m.name) == "person"]
            mentioned += curr_people
        
        return mentioned
        
            
    def mentioned_locations(self):
        """ 
        Return list of locations mentioned thus far in story
        """
        mentioned = []
        for k, v in self.history.items():
            if type(v.target) == list:
                ents = [self.get_entity_by_name(t)for t in v.target]
                if all([self.ent_map.get(e) == "location" for e in ents]):
                    mentioned += ents
            else:
                ent = self.get_entity_by_name(v.target)
                if self.ent_map.get(ent.name) == "location":
                    mentioned += [ent]
        return mentioned               
            
        
        
        

    def forget(self):
        """
        Freshly starts the holding-holder hierarchy in game. Nothing is holding anything, nothing is being held.
        All previous information about what is holding what is deleted.
        """
        self.timestep = 0
        self.question_list.forget()
        for entity in self.entities:
            entity.reset()
        
        # erase event histories   
        self.history = {}
        self.q_history = {}
        self.diff_props = defaultdict(list)

    def add_entity(self, entity):
        self.entities.add(entity)

    def add_known_item(self, a, b, kind = None, match_location = None, 
                       only_world_record: bool = False):
        """
        Adds new information about an entity, that should be known the reader of a bAbI story (if the reader reads well)
        Usually a will be some entity and b will be the entity that is known to hold a (but not necessarily).
        """
        if not only_world_record:
            self.question_list.add_known_item(a, b, match_location=match_location)
        self.diff_props[self.timestep] = list(set(self.diff_props[self.timestep]).union(set(prop_factory(a, b))))
        

    def remove_known_item(self, a, b, only_world_record: bool = False, belief_prob: float = 0.0):
        """
        Adds new information about an entity, that should be known the reader of a bAbI story (if the reader reads well)
        Usually a will be some entity and b will be the entity that is known to  no longer hold a (but not necessarily).
        """
        if not only_world_record:
            self.question_list.remove_known_item(a, b)
        self.diff_props[self.timestep] += prop_factory(a, b, belief=belief_prob)
        
        
    def update_all_neg_poss(self, a, pos_poss = None, only_world_record: bool = True,
                       belief_prob: float = 0.0):
        if not pos_poss:
            pos_poss = [a.holder]
        
        mentioned_people = self.mentioned_people()
        for p in mentioned_people:
            if not p in pos_poss:
                self.remove_known_item(a, p, only_world_record=only_world_record, 
                                             belief_prob=belief_prob)
        
    
    def update_all_neg(self, a, pos_locs = None, only_world_record: bool = True,
                       belief_prob: float = 0.0):
        if not pos_locs:
            pos_locs = [a.holder]
        
        mentioned_locs = self.mentioned_locations()
        for l in mentioned_locs:
            if not l in pos_locs:
                self.remove_known_item(a, l, only_world_record=only_world_record, 
                                             belief_prob=belief_prob)
        for object in a.holds:
            # record that object isn't at any other location
            for l in mentioned_locs:
                if not l in pos_locs:
                    self.remove_known_item(object, l, only_world_record=only_world_record,
                                                 belief_prob=belief_prob)

    def can_act(self):
        """
        Can any action be performed. Note it in the usual case we have at least one person and two locations,
        and move actions are allowed - in that case some action can always be performed.
        """
        return self.action_list.can_act()

    def make_action(self):
        actions = self.action_list.make_action()
        return actions

    def can_ask(self):
        """
        Can any question be asked. A question can only be asked if there is some information known to the reader that
        the question can be asked about.
        """
        return self.question_list.can_ask()

    
    def ask(self, exhaustive: bool = False):
        """
        Ask question or questions valid in current game state.

        Parameters
        ----------
        exhaustive : bool, optional
            Ask all possible questions if True, otherwise- ask single question.

        Returns
        -------
        Questions : list of questions
        Answers : list of associated answers.

        """
        if exhaustive:
            return self.ask_all_questions()
        else:
            question, ans = self.ask_question()
            return [question], [ans]
        
        

    def ask_question(self):
        self.timestep += 1
        return self.question_list.ask_question()
    
    def ask_all_questions(self):
        all_qs = self.question_list.ask_all_valid_questions()
        # self.timestep += len(all_qs)
        return zip(*all_qs)

    def get_entity_by_name(self, name):
        return [entity for entity in self.entities if entity.name == name][0]

    def get_action_by_kind(self, kind):
        return [action for action in self.action_list.actions if action.kind == kind][0]

    def get_question_by_kind(self, kind):
        return [question for question in self.question_list.questions if question.kind == kind][0]
    
    def curr_support_idxs(self):
        """ 
        Return current sentence indices supporting this event:
            current timestep, and previous if this event is a coref.
        """
        t = self.timestep
        curr_event = self.history[t]
        
        # if coref event, add previous event as support (needed to resolve co-ref)
        if curr_event.is_coref:
            curr_support = set([t-1, t])
        else:
            curr_support = set([t])
        
        return curr_support


    def to_dec_story(self, babi_story: List[str]) -> DECStory:
        """ 
        Export events to DEC story format.
        """
        history = {}
        for event in self.history.values():
            history[event.timestep] = from_world_event(event)
            
        for q_event in self.q_history.values():
            history[q_event.timestep] = from_world_q_event(q_event)
        
        _, evs = zip(*sorted(history.items(), key=lambda x: x[0]))
        
        dec_story = DECStory(seed=self.current_seed,
                             task=self.params.name,
                             events=list(evs),
                             coref_map=self.params.entity_coreference_map,
                             babi_story=babi_story,
                             ie_answers={},
                             ie_s_facts={})
        return dec_story
    

