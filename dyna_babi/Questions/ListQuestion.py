from collections import defaultdict 
from .Question import Question
from ..helpers.utils import choice_np, supp_facts_str
from ..helpers.event import Event, QuestionEvent, QuestionType

class ListQuestion(Question):
    def __init__(self, persons, world):
        super().__init__(world, "What is {} carrying?\t{}", "list", {})
        self.persons = persons
        self.possesion_facts_index = defaultdict(lambda: set())

    def add_known_item(self, a, b, t=None, match_location=None):
        """
        Adds the pair of a,b to self.known_items - if tracking of the type of a,b is relevant for this question
        """
        if a.kind == "object" and b.kind == "person":
            supp_idxs = self.world.curr_support_idxs()
            self.possesion_facts_index[b.name].update(supp_idxs)
            if b in self.known_items:
                self.known_items[b].add(a)
            else:
                self.known_items[b] = {a}
            

    def remove_known_item(self, a, b):
        """
        Removes the pair of a,b from self.known_items - if tracking of the type of a,b is relevant for this question
        """
        if a.kind == "object" and b.kind == "person":
            supp_idxs = self.world.curr_support_idxs()
            self.possesion_facts_index[b.name].update(supp_idxs)
            if b in self.known_items:
                self.known_items[b].discard(a)
            if len(self.known_items[b]) == 0:
                self.known_items.pop(b, None)

    def forget(self):
        self.known_items.clear()
        self.possesion_facts_index = defaultdict(lambda: set())

    def is_valid(self):
        if self.world.params.exhaustive:
            return True
        else:
            return self.known_items
    
    def all_valid(self):
        """ 
        Return all valid people for which ask specific can be called.
        """
        return sorted(list(self.persons), key=lambda x: x.name)
    
    def supporting_facts_for_target(self, target): 
        return sorted(list(self.possesion_facts_index[target.name]))
    

    def ask(self, max_support: bool = False):
        """
        Asks a question with uniform distribution over self.persons.
        """
        if max_support:
            # ask the q requiring most supporting facts
            facts_per_item = sorted([(len(ListQuestion.supporting_facts_for_target(item)), item) 
                              for item in self.known_items], key=lambda x: x[0])
            max_item = facts_per_item[-1]
            _, b = max_item
                
        else:
            b = choice_np(list(self.persons))
            
        supp_idxs = self.supporting_facts_for_target(b)
        supp_facts_idxs = supp_facts_str(supp_idxs)
        
        if b not in self.known_items:
            objects_str = "nothing"
            objects = []
        else:
            objects = list(self.known_items[b])
            objects.sort(key=lambda object: object.name)
            objects_str = "".join([object.name + ", " for object in objects])[:-2]

        sentence = self.template.format(b.name, objects_str) + supp_facts_idxs
        graph_rep = [[["Q_LIST", ""], ["P", b.name]]]
        answer = objects_str
        self.update_q_history(b.name, [object.name for object in objects], supp_idxs=supp_idxs)
        return [sentence, graph_rep], answer
    
    def update_q_history(self, person, answer, supp_idxs):
        q = QuestionEvent(kind=QuestionType(self.kind),
                     timestep=self.world.timestep,
                     source=person,
                     target=[answer],
                     supporting_facts=supp_idxs)
        self.world.q_history[self.world.timestep] = q

    def ask_specific(self, person):
        """
        Asks a question about a specific entity.
        """
        b = person
        if b not in self.known_items:
            objects_str = "nothing"
            objects = []
        else:
            objects = list(self.known_items[b])
            objects.sort(key=lambda object: object.name)
            objects_str = "".join([object.name + "," for object in objects])[:-1]
        
        supp_idxs = self.supporting_facts_for_target(b)
        supp_facts_idxs = supp_facts_str(supp_idxs)
        
        
        sentence = self.template.format(b.name, objects_str) + supp_facts_idxs
        graph_rep = [[["Q_LIST", ""], ["P", b.name]]]
        answer = objects_str
        self.update_q_history(b.name, [object.name for object in objects], supp_idxs=supp_idxs)
        return [sentence, graph_rep], answer
