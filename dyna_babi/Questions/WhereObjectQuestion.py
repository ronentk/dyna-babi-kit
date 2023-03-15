from collections import defaultdict 
from .Question import Question
from .WherePersonQuestion import WherePersonQuestion
from ..helpers.utils import choice_np, supp_facts_str
from ..helpers.event import find_all_loc_supports
from ..helpers.event import QuestionEvent, QuestionType


class WhereObjectQuestion(Question):
    def __init__(self, world):
        super().__init__(world, "Where is the {}?\t{}", "where_object")
        self.known_people = set()
        self.location_facts_index = defaultdict(lambda: set())
        self.possesion_facts_index = defaultdict(lambda: set())
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
            # shouldn't it be broadcast to all questions as )obj, loc) pair like below?
            for object in a.holds:
                self.known_items.add(object)
                self.world.add_known_item(object, b, only_world_record=True)
                self.world.update_all_neg(object, [b])
                self.location_facts_index[object.name] = set().union( *[self.possesion_facts_index[object.name], self.location_facts_index[a.name]])
                if match_location:
                    self.implicit_supports[object.name].add(t)
        
        if a.kind == "object" and b.kind == "location":
            # move (for other objects held by person)
            # TODO check if match locations should also affect
            p = a.holder
            assert(p.kind == "person")
            self.known_items.add(a)
            self.world.add_known_item(a, b, only_world_record=True)
            self.world.update_all_neg(a, [b])
            self.location_facts_index[a.name] = set().union( *[self.possesion_facts_index[a.name], self.location_facts_index[p.name]])
        
        if a.kind == "object" and b.kind == "person":
            # give, grab
            self.possesion_facts_index[a.name] = curr_support
            if b in self.known_people:
                self.known_items.add(a)
                self.world.add_known_item(a, b.holder, only_world_record=True)
                self.world.update_all_neg(a, [b.holder])
                self.location_facts_index[a.name] = set().union( *[self.possesion_facts_index[a.name], self.location_facts_index[b.name]])
                
        
    def remove_known_item(self, a, b):
        """
        Removes the pair of a,b from self.known_items - if tracking of the type of a,b is relevant for this question
        """
        if a.kind == "person" and b.kind == "location":
            self.known_people.discard(a)
            self.location_facts_index[a.name] = []
            for object in self.known_items.copy():
                if object.holder == a:
                    self.known_items.discard(object)
                    self.world.remove_known_item(a, b, only_world_record=True)
                    self.location_facts_index[object.name] = []
        if a.kind == "object" and b.kind == "person":
            # drop or give
            prev_pos_facts = self.possesion_facts_index[a.name]
            
            # remove previous possesion facts only, not move <?>
            prev_pos_facts = set([t for t in prev_pos_facts if self.world.history[t].kind in ["give", "grab", "drop"]])
            if not self.world.history[self.world.timestep].is_all_act:
                # if this is a drop/give all action, we don't want to erase old possession
                # fact, since the drop all sentence only makes implicit ref. to this object
                old_facts = prev_pos_facts if not prev_pos_facts == self.world.curr_support_idxs() else set()
            else:
                old_facts = set()
                
            self.possesion_facts_index[a.name] = self.world.curr_support_idxs()
            self.location_facts_index[a.name] = set().union( *[self.possesion_facts_index[a.name], self.location_facts_index[a.name]]) - old_facts

    def forget(self):
        self.known_people.clear()
        self.known_items.clear()
        self.location_facts_index = defaultdict(lambda: set())
        self.possesion_facts_index = defaultdict(lambda: set())
        self.implicit_supports = defaultdict(lambda: set())

    def locate(self, object):
        """
        :return: The current location of object
        """
        if object.holder.kind == "location" or object.holder.kind is None:
            return object.holder
        else:
            return self.locate(object.holder)

    def is_valid(self):
        return self.known_items
    
    def all_valid(self):
        return sorted(list(self.known_items), key=lambda x: x.name)
    
    def supporting_facts_for_target(self, target): 
        return sorted(list(set().union( *[self.possesion_facts_index[target.name], self.location_facts_index[target.name]])))
    

    
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

        sentence = self.template.format(a.name, self.locate(a).name) + supp_facts_idxs
        graph_rep = [[["Q_WHERE_O", ""], ["O", a.name], ["L", self.locate(a).name]]]
        answer = self.locate(a).name
        self.update_q_history(a.name, answer, supp_idxs=supp_idxs)
        return [sentence, graph_rep], answer
    
    def update_q_history(self, obj, location, supp_idxs):
        q = QuestionEvent(kind=QuestionType(self.kind),
                     timestep=self.world.timestep,
                     source=obj,
                     target=[location],
                     supporting_facts=supp_idxs,
                     implicit_facts=set(self.implicit_supports[obj]).intersection(supp_idxs))
        self.world.q_history[self.world.timestep] = q

    def ask_specific(self, object):
        """
        Asks a question about a specific entity.
        """
        known_object = [item for item in self.known_items if item == object][0]
        
        # record supporting facts for q on chosen item
        supp_idxs = self.supporting_facts_for_target(known_object)
        supp_facts_idxs = supp_facts_str(supp_idxs)
        
        sentence = self.template.format(known_object.name, self.locate(known_object).name) + supp_facts_idxs
        graph_rep = [[["Q_WHERE_O", ""], ["O", known_object.name], ["L", self.locate(known_object).name]]]
        answer = self.locate(known_object).name
        self.update_q_history(known_object.name, answer, supp_idxs=supp_idxs)
        return [sentence, graph_rep], answer
