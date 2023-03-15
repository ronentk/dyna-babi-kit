from collections import defaultdict 

from .Question import Question
from ..helpers.utils import choice_np, supp_facts_str
from ..helpers.event import Event, QuestionEvent, QuestionType

class WherePersonQuestion(Question):
    def __init__(self, world):
        super().__init__(world, "Where is {}?\t{}", "where_person")
        self.location_facts_index = defaultdict(lambda: set())
        self.implicit_supports = defaultdict(lambda: set())

    def add_known_item(self, a, b, t=None, match_location=None):
        """
        Adds the pair of a,b to self.known_items - if tracking of the type of a,b is relevant for this question
        """
        curr_event = self.world.history[t]
        
        
        
        # if coref event, add previous event as support (needed to resolve co-ref)
        if curr_event.is_coref:
            curr_support = set([t-1, t])
        else:
            curr_support = set([t])
            
        if a.kind == "person" and b.kind == "location":
            if match_location:
                # called through match locations: we know person a + carried objects
                # location supported by same facts as match_location for object, so we use
                # where_object location tracking
                # TODO hacky to rely on other question here as they then must be both present in question list
                wo_q = self.world.get_question_by_kind("where_object")
                curr_support = wo_q.location_facts_index[match_location.name].union(curr_support)
                self.implicit_supports[a.name].add(t)
                
            self.known_items.add(a)
            self.location_facts_index[a.name] = curr_support

    def remove_known_item(self, a, b):
        """
        Removes the pair of a,b from self.known_items - if tracking of the type of a,b is relevant for this question
        """
        if a.kind == "person" and b.kind == "location":
            self.known_items.discard(a)
            self.location_facts_index[a.name] = []

    def forget(self):
        self.known_items.clear()
        self.location_facts_index = defaultdict(lambda: set())
        self.implicit_supports = defaultdict(lambda: set())

    def is_valid(self):
        return self.known_items
    
    def all_valid(self):
        return sorted(list(self.known_items), key=lambda x: x.name)
    
    
    def supporting_facts_for_target(self, target, world = None):
        """ 
        Return supporting facts for a question on target
        """
        assert (target.name in self.location_facts_index)
        return sorted(list(self.location_facts_index[target.name]))
    

    def ask(self, max_support: bool = False):
        """
        Asks a question with uniform distribution over self.known_items.
        """
        if max_support:
            # ask the q requiring most supporting facts
            facts_per_item = sorted([(len(self.supporting_facts_for_target(item)), item) 
                              for item in self.known_items], key=lambda x: x[0])
            max_item = facts_per_item[-1]
            _, a = max_item
                
        else:
            a = choice_np(list(self.known_items))

        supp_idxs = self.supporting_facts_for_target(a)
        supp_facts_idxs = supp_facts_str(supp_idxs)
        
        sentence = self.template.format(a.name, a.holder.name) + supp_facts_idxs
        graph_rep = [[["Q_WHERE_P", ""], ["P", a.name], ["L", a.holder.name]]]
        answer = a.holder.name
        self.update_q_history(a.name, answer, supp_idxs=supp_idxs)
        return [sentence, graph_rep], answer
    
    def update_q_history(self, person, loc, supp_idxs):
        q = QuestionEvent(kind=QuestionType(self.kind),
                     timestep=self.world.timestep,
                     source=person,
                     target=[loc],
                     supporting_facts=supp_idxs,
                     implicit_facts=set(self.implicit_supports[person]).intersection(supp_idxs))
        self.world.q_history[self.world.timestep] = q

    def ask_specific(self, person):
        """
        Asks a question about a specific entity.
        """
        known_person = [item for item in self.known_items if item == person][0]
        
        # record supporting facts for q on chosen item
        supp_idxs = self.supporting_facts_for_target(known_person)
        supp_facts_idxs = supp_facts_str(supp_idxs)

        sentence = self.template.format(known_person.name, known_person.holder.name) + supp_facts_idxs
        graph_rep = [[["Q_WHERE_P", ""], ["P", known_person.name], ["L", known_person.holder.name]]]
        answer = known_person.holder.name
        self.update_q_history(known_person.name, answer, supp_idxs=supp_idxs)
        return [sentence, graph_rep], answer
