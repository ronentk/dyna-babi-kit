from collections import defaultdict 

from .Question import Question
from ..Actions.Action import ActionType
from ..helpers.event import Event, QuestionEvent, QuestionType

from ..helpers.utils import choice_np, supp_facts_str


class WhereWasObjectQuestion(Question):
    def __init__(self, world):
        super().__init__(world, "Where was the {} before the {}?\t{}", "where_was_object", {})
        self.known_people = set()
        self.location_facts_index = defaultdict(lambda: set())
        self.possesion_facts_index = defaultdict(lambda: set())
        self.history_facts_index = defaultdict(lambda: list())
        self.implicit_supports = defaultdict(lambda: set())

    def add_known_item(self, a, b, t=None, match_location=None):
        """
        Adds the pair of a,b to self.known_items / self.known_people -
        if tracking of the type of a,b is relevant for this question
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
                # location supported by same facts as match_location
                curr_support = self.location_facts_index[match_location.name].union(curr_support)
                
                
            self.known_people.add(a)
            self.location_facts_index[a.name] = curr_support
            for object in a.holds:
                obj_support = curr_support.union(self.possesion_facts_index[object.name])
                if object in self.known_items:
                    if self.known_items[object][-1] != b:
                        self.known_items[object].append(b)
                        self.history_facts_index[object.name].append((obj_support, b.name, True))
                else:
                    self.known_items[object] = [a.holder]
                    self.history_facts_index[object.name] = [(obj_support, b.name, False)]
                if match_location:
                    self.implicit_supports[object.name].add(t)
        
        if a.kind == "object" and b.kind == "location":
            p = a.holder
            assert(p.kind == "person")
            curr_support = set().union(*[self.possesion_facts_index[a.name], self.location_facts_index[p.name]])
            if a in self.known_items:
                if self.known_items[a][-1] != b:
                    self.known_items[a].append(b)
                    self.history_facts_index[a.name].append((curr_support, b.name, True))
            else:
                self.known_items[a] = [b]
                self.history_facts_index[a.name] = [(self.location_facts_index[a.name], b.name, False)]
        
        if a.kind == "object" and b.kind == "person":
            self.possesion_facts_index[a.name] = curr_support
            b_support = self.possesion_facts_index[a.name].union(self.location_facts_index[b.name])
            if b in self.known_people:
                if a in self.known_items:
                    
                    if self.known_items[a][-1] != b.holder:
                        self.known_items[a].append(b.holder)
                        self.history_facts_index[a.name].append((b_support, b.holder.name, True))
                    else:
                        self.history_facts_index[a.name].append((b_support, b.holder.name, False))
                        
                else:
                    self.known_items[a] = [b.holder]
                    self.history_facts_index[a.name] = [(b_support, b.holder.name, False)]

    def remove_known_item(self, a, b):
        """
        Removes the pair of a,b from self.known_items - if tracking of the type of a,b is relevant for this question
        """
        if a.kind == "person" and b.kind == "location":
            self.known_people.discard(a)
            self.location_facts_index[a.name] = set()
            for object in self.known_items.copy():
                if object.holder == a:
                    self.known_items[object].append(None)
                    self.history_facts_index[a.name].append(None)

    def forget(self):
        self.known_people.clear()
        self.known_items.clear()
        self.location_facts_index = defaultdict(lambda: set())
        self.possesion_facts_index = defaultdict(lambda: set())
        self.history_facts_index = defaultdict(lambda: list())
        self.implicit_supports = defaultdict(lambda: set())

    def locate(self, object):
        """
        :return: The current location of object
        """
        if object.holder.kind == "location" or object.holder.kind is None:
            return object.holder
        else:
            return self.locate(object.holder)
        
    def supporting_facts_for_target(self, obj, loc):
        """ 
        Return supporting facts for a question of type: where was <obj> before <loc>
        """
        assert(obj.name in self.history_facts_index)
        supp_facts = set()
        obj_hist = self.history_facts_index[obj.name]
        
        # find event when q became valid
        for i, (ev_supp_idxs, loc_name, valid_q) in enumerate(obj_hist):
            if loc_name == loc.name and valid_q:
                supp_facts = supp_facts.union(ev_supp_idxs)
                break
        
        assert(i > 0)
        prev_ev_idxs, _, _ = obj_hist[i-1]
        supp_facts = supp_facts.union(prev_ev_idxs)
        return sorted(list(supp_facts))
        
        
    

    def is_valid(self):
        """
        True iff at least one object's current location is known, as well as that object's previous location
        """
        for object in self.known_items:
            count = 0
            for location in self.known_items[object]:
                if location is not None:
                    count += 1
                else:
                    count = 0
                if count > 1:
                    return True
        return False

    def all_valid(self):
        """ 
        Return list of all triples parametereizing valid questions
        """
        triples = []
        triples_objs = [] # hacky since no dict access to objs by name
        for object in self.known_items:
            locations = self.known_items[object][::-1]
            count = 0
            location2 = None
            for location1 in locations:
                if location1 is not None:
                    count += 1
                else:
                    count = 0
                if count > 1:
                    for triple in triples:
                        if triple[1] == location2.name:
                            break
                    else:
                        triples.append((object.name, location2.name, location1.name))
                        triples_objs.append((object, location2))
                location2 = location1
        
        return sorted(triples_objs, key=lambda x: f"{x[0].name}_{x[1].name}")
    
    # def all_valid_questions(self):
    #     """ 
    #     Ask all valid questions at current world state.
    #     """
    #     qs = []
    #     for obj, loc2, loc1 in self.all_valid():
    #         q = self.ask_specific(obj, loc2)
    #         qs.append(q)
    #     return qs
            
        

    def ask(self, max_support: bool = False):
        """
        Asks a question with uniform distribution over self.known_items.
        """
        triples = []
        triples_objs = [] # hacky since no dict access to objs by name
        for object in self.known_items:
            locations = self.known_items[object][::-1]
            count = 0
            location2 = None
            for location1 in locations:
                if location1 is not None:
                    count += 1
                else:
                    count = 0
                if count > 1:
                    for triple in triples:
                        if triple[1] == location2.name:
                            break
                    else:
                        triples.append((object.name, location2.name, location1.name))
                        triples_objs.append((object, location2, location1))
                location2 = location1
                
                
        if max_support:
            # ask the q requiring most supporting facts
            len_max_supp = 0
            chosen_supp_idxs = []
            for obj, loc_2, loc_1 in triples_objs:
                supp_idxs = self.supporting_facts_for_target(obj, loc_2)
                if len(supp_idxs) > len_max_supp:
                    len_max_supp = len(supp_idxs)
                    triple = (obj.name, loc_2.name, loc_1.name)
                    chosen_supp_idxs = supp_idxs
                
        else:
            triple = choice_np(triples)
            # get supporting facts idxs
            obj, loc_2, _ = triples_objs[triples.index(triple)]
            chosen_supp_idxs = self.supporting_facts_for_target(obj, loc_2)

        supp_facts_idxs = supp_facts_str(chosen_supp_idxs)
        
        sentence = self.template.format(*triple) + supp_facts_idxs
        graph_rep = [[["BEFORE_L", ""], ["L", triple[2]]],
                [["AFTER_L", ""], ["L", triple[1]]],
                [["Q_WHERE_WAS_O", ""], ["O", triple[0]], ["BEFORE_L", ""], ["AFTER_L", ""]]]
        answer = triple[2]
        self.update_q_history(*triple, supp_idxs=chosen_supp_idxs)
        return [sentence, graph_rep], answer
    
    
    def update_q_history(self, obj, after_loc, before_loc, supp_idxs):
        q = QuestionEvent(kind=QuestionType(self.kind),
                     timestep=self.world.timestep,
                     source=obj,
                     ternary=after_loc,
                     target=[before_loc],
                     supporting_facts=supp_idxs,
                     implicit_facts=set(self.implicit_supports[obj]).intersection(supp_idxs))
        self.world.q_history[self.world.timestep] = q

    def ask_specific(self, object, location):
        """
        Asks a question about specific entities.
        """
        if object not in self.known_items.keys():
            raise ValueError
            
        locations = self.known_items[object]
        locations = locations[::-1]

        count = 0
        location2 = None
        for location1 in locations:
            if location1 is not None:
                count += 1
            else:
                count = 0
            if count > 1:
                if location2 == location:
                    triple = (object.name, location2.name, location1.name)
                    supp_idxs = self.supporting_facts_for_target(object, location2)
                    supp_idxs_str = supp_facts_str(supp_idxs)

                    sentence = self.template.format(*triple) + supp_idxs_str
                    graph_rep = [[["BEFORE_L", ""], ["O", triple[2]]],
                            [["AFTER_L", ""], ["O", triple[1]]],
                            [["Q_WHERE_WAS_O", ""], ["O", triple[0]], ["BEFORE_L", ""], ["AFTER_L", ""]]]
                    answer = triple[2]
                    self.update_q_history(*triple, supp_idxs=supp_idxs)
                    return [sentence, graph_rep], answer

            location2 = location1
        # logically, should never reach this line
        return None, None
        raise ValueError