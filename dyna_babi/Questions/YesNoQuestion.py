from collections import defaultdict 
from .Question import Question
import numpy.random as random
from ..helpers.utils import choice_np, supp_facts_str
from ..helpers.event import QuestionEvent, QuestionType


class YesNoQuestion(Question):
    def __init__(self, locations, world):
        super().__init__(world, "Is {} in the {}?\t{}", "yes_no")
        self.locations = locations
        self.known_no = {}
        self.known_maybe = {}
        self.known_facts_index = defaultdict(lambda: set())
        self.known_no_index = defaultdict(lambda: set())
        self.known_maybe_index = defaultdict(lambda: set())
        self.implicit_supports = defaultdict(lambda: set())
        

    def add_known_item(self, a, b, t=None, match_location=None):
        """
        Adds the pair of a,b to self.known_items / self.known_maybe or removes it from self.known_no-
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
                # called through match locations: we know person a + carried objects")
                # # location supported by same facts as match_location
                wo_q = self.world.get_question_by_kind("where_object")
                curr_support = wo_q.location_facts_index[match_location.name].union(curr_support)
                self.implicit_supports[a.name].add(t)
                
            self.known_items.add(a)
            self.known_facts_index[a.name] = curr_support
            self.known_no.pop(a, None)
            self.known_no_index[a.name].intersection_update(set())
            self.known_maybe.pop(a, None)
            self.known_maybe_index[a.name].intersection_update(set())
        if a.kind == "person" and b.kind == "indef_location":
            self.known_maybe[a] = b
            self.known_maybe_index[a.name] = curr_support
            self.known_items.discard(a)
            self.known_facts_index[a.name].intersection_update(set())
            self.known_no.pop(a, None)
            self.known_no_index[a.name].intersection_update(set())

    def remove_known_item(self, a, b):
        """
        Removes the pair of a,b from self.known_items / self.known_maybe or adds it to self.known_no-
        if tracking of the type of a,b is relevant for this question
        """
        t = self.world.timestep
        curr_event = self.world.history[t]
        
        # if coref event, add previous event as support (needed to resolve co-ref)
        if curr_event.is_coref:
            curr_support = set([t-1, t])
        else:
            curr_support = set([t])
            
        if a.kind == "person" and b.kind == "location":
            self.known_items.discard(a)
            self.known_facts_index[a.name].intersection_update(set())
            self.known_maybe.pop(a, None)
            self.known_maybe_index[a.name].intersection_update(set())
            self.known_no[a] = b
            self.known_no_index[a.name] = curr_support

    def forget(self):
        self.known_no.clear()
        self.known_maybe.clear()
        self.known_items.clear()
        self.known_facts_index = defaultdict(lambda: set())
        self.known_no_index = defaultdict(lambda: set())
        self.known_maybe_index = defaultdict(lambda: set())
        self.implicit_supports = defaultdict(lambda: set())

    def is_valid(self):
        return self.known_items or self.known_no or self.known_maybe
    
    
    def all_valid(self):
        """ 
        Return all valid (obj, location) pairs for which ask specific can be called.
        Only calling a subset of possible questions- is person at known/indef location
        (answered by yes/maybe), and is person at the loction it's known not to be at
        answered by `no`.
        """
        all_pairs = []
        
        # for known and indef, ask about the mentioned location
        for item in set.union(self.known_items, self.known_maybe):
            all_pairs.append((item, item.holder))
        
        # ask about locations person known not to be at
        for item, known_no_loc in self.known_no.items():
            all_pairs.append((item, known_no_loc))
            
        if self.world.params.extra_exh_yes_no:
            # for each known/maybe person also ask about all locations not at
            for item in sorted(set.union(self.known_items, self.known_maybe), 
                               key=lambda x: f"{x.name}"):
                not_at_locs = sorted([l for l in self.locations if l != item.holder],
                                     key=lambda x: f"{x.name}")
                for loc in not_at_locs:
                    all_pairs.append((item, loc))
        else: 
            # for each known/maybe person also ask about random location
            for item in sorted(set.union(self.known_items, self.known_maybe), 
                               key=lambda x: f"{x.name}"):
                location = choice_np(self.locations)
                all_pairs.append((item, location))
            
        return sorted(all_pairs, key=lambda x: f"{x[0].name}_{x[1].name}")
        
    

    
    @classmethod
    def supporting_facts_for_target(cls, target):
        """ 
        Return supporting facts for a question on target person
        """        
        facts = []
        if target.indef_loc:
            # if not definitely known, return the event with the indefinite info
            facts = target.indef_loc_ev
        else:
            # otherwise return events supporting knowledge of location
            facts = target.last_known_loc
        return sorted(list(set([f.timestep for f in facts])))

    def ask(self, max_support: bool = False):
        """
        Asks a question with uniform distribution over self.known_items + self.known_no + self.known_maybe.
        Max support not supported currently.
        """
        sentence_elements = ()
        while not sentence_elements:
            p = random.choice([0, 1, 2])
            if p == 0:
                if self.known_items:
                    a = choice_np(list(self.known_items))
                    sentence_elements = (a.name, a.holder.name, "yes")
                    supp_idxs = self.known_facts_index[a.name]
            elif p == 1:
                if random.choice([0, 1]) == 0:
                    if self.known_items:
                        a = choice_np(list(self.known_items))
                        location = choice_np(self.locations)
                        while location.name == a.holder.name:
                            location = choice_np(self.locations)
                        sentence_elements = (a.name, location.name, "no")
                        supp_idxs = self.known_facts_index[a.name]
                else:
                    if self.known_no:
                        a = choice_np(list(self.known_no))
                        sentence_elements = (a.name, self.known_no[a].name, "no")
                        supp_idxs = self.known_no_index[a.name]
            elif p == 2:
                if self.known_maybe:
                    a = choice_np(list(self.known_maybe))
                    location = choice_np(self.locations)
                    maybe_locations = self.known_maybe[a][1:]
                    if location in maybe_locations:
                        sentence_elements = (a.name, location.name, "maybe")
                    else:
                        sentence_elements = (a.name, location.name, "no")
                    supp_idxs = self.known_maybe_index[a.name]
        
        supp_facts_idxs = supp_facts_str(supp_idxs)
        sentence = self.template.format(*sentence_elements) + supp_facts_idxs
        graph_rep = [[["Q_YES_NO", ""], ["P", sentence_elements[0]], ["L", sentence_elements[1]]]]
        self.update_q_history(*sentence_elements, supp_idxs=list(supp_idxs))
        return [sentence, graph_rep], sentence_elements[2]
    
    def update_q_history(self, person, location, answer, supp_idxs):
        q = QuestionEvent(kind=QuestionType(self.kind),
                     timestep=self.world.timestep,
                     source=person,
                     ternary=location,
                     target=[answer],
                     supporting_facts=supp_idxs,
                     implicit_facts=set(self.implicit_supports[person]).intersection(supp_idxs))
        self.world.q_history[self.world.timestep] = q
    
    def ask_specific(self, person, location):
        """
        Asks a question about specific entities.
        """
        sentence_elements = ()
        answer = ""
        if person in self.known_items:
            if person.holder == location:
                sentence_elements = (person.name, location.name, "yes")
                answer = "yes"
                supp_idxs = self.known_facts_index[person.name]
            else:
                sentence_elements = (person.name, location.name, "no")
                answer = "no"
                supp_idxs = self.known_facts_index[person.name]
        elif person in self.known_no and self.known_no[person] == location:
            sentence_elements = (person.name, location.name, "no")
            answer = "no"
            supp_idxs = self.known_no_index[person.name]
            
        elif person in self.known_maybe:
            if location in self.known_maybe[person]:
                sentence_elements = (person.name, location.name, "maybe")
                answer = "maybe"
            else:
                sentence_elements = (person.name, location.name, "no")
                answer = "no"
            supp_idxs = self.known_maybe_index[person.name]

        try:
            supp_facts_idxs = supp_facts_str(supp_idxs)
            sentence = self.template.format(*sentence_elements) + supp_facts_idxs
        except:
            return None, None
        self.update_q_history(*sentence_elements, supp_idxs)
        graph_rep = [[["Q_YES_NO", ""], ["P", sentence_elements[0]], ["L", sentence_elements[1]]]]
        return [sentence, graph_rep], answer
