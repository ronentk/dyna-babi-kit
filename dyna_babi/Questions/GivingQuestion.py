from .Question import Question
import numpy.random as random
from ..helpers.utils import choice_np, supp_facts_str
from ..Actions.Action import ActionType
from ..helpers.event import QuestionEvent, QuestionType

class GivingQuestion(Question):
    def __init__(self, world):
        super().__init__(world, "", "giving")
        self.known_objects = {}
        self.known_passers = {}
        self.known_receivers = {}
        self.known_passings = {}

    def add_known_item(self, a, b, t=None, match_location=None):
        """
        Adds the pair of a,b to self.known_objects, self.known_passers, self.known_receivers and self.known_passings
        if tracking of the type of a,b is relevant for this question
        """
        if a.kind == "give_triple" and b.kind == "person":
            self.known_objects[a.object] = a
            self.known_passers[(a.person1, a.object)] = a
            self.known_receivers[(a.object, a.person2)] = a
            self.known_passings[(a.person1, a.person2)] = a

    def remove_known_item(self, a, b):
        pass

    def forget(self):
        self.known_objects.clear()
        self.known_passers.clear()
        self.known_passings.clear()
        self.known_receivers.clear()
        self.known_items.clear()

    def is_valid(self):
        return self.known_objects
    
    def all_valid(self):
        """ 
        Asking only one kind of question here as all are equally easy
        """
        known_passing = sorted(list(self.known_passings.values()), key=lambda x: f"{x.person1.name}_{x.person2.name}_{x.object.name}")
        return [(p.person1, p.person2, p.object, "what_give") for p in known_passing]
    
    @classmethod
    def supporting_facts_for_target(cls, target):
        """ 
        Return supporting facts for a question on target
        """
        facts = []
        give_evs = sorted([e for t, e in target.history.items() if e.kind == ActionType.GIVE.value])
        if give_evs:
            facts += [give_evs[-1]]
        
        return sorted(list(set([f.timestep for f in facts])))

    def ask(self, max_support: bool = False):
        """
        Asks a question with uniform distribution over the known tuples of information the question holds.
        
        Max support  == True has no effect since all giving questions require only one supp fact
        """
        known_object = choice_np(list(self.known_objects.values()))
        known_passer = choice_np(list(self.known_passers.values()))
        known_receiver = choice_np(list(self.known_receivers.values()))
        known_passing = choice_np(list(self.known_passings.values()))
        # each element in sentences_triples_answers_surfaces is a list containing:
        # a bAbI giving question (sentence), the triple of entities tha partake in the giving,
        # the answer to the question and a graph_rep decription of the sentence (surface)
        t1 = (known_receiver.object.name, known_receiver.person2.name, known_receiver.person1.name)
        t2 = (known_object.object.name, None, known_object.person1.name)
        t3 = (known_object.object.name, None, known_object.person2.name)
        t4 = (known_passer.person1.name, known_passer.object.name, known_passer.person2.name)
        t5 = (known_passing.person1.name, known_passing.person2.name, known_passing.object.name)
        sentences_triples_answers_surfaces = [["Who gave the {} to {}?\t{}".format(*t1), known_receiver, known_receiver.person1.name, "gave_to", t1],
                                              ["Who gave the {}?\t{}".format(known_object.object.name, known_object.person1.name), known_object, known_object.person1.name, "gave", t2],
                                              ["Who received the {}?\t{}".format(known_object.object.name, known_object.person2.name), known_object, known_object.person2.name, "received", t3],
                                              ["Who did {} give the {} to?\t{}".format(*t4), known_passer, known_passer.person2.name, "who_give", t4],
                                              ["What did {} give to {}?\t{}".format(*t5), known_passing, known_passing.object.name, "what_give", t5]]

        sentence, triple, answer, surface, triple_names = choice_np(sentences_triples_answers_surfaces)
        
        
        supp_idxs = GivingQuestion.supporting_facts_for_target(triple.object)
        supp_facts_idxs = supp_facts_str(supp_idxs)
        
        self.update_q_history(*triple_names, supp_idxs=supp_idxs, q_sub_type=surface)
        
        sentence = sentence + supp_facts_idxs
        
        graph_rep = [[["SOURCE_P", ""], ["P", triple.person1]],
                [["TARGET_P", ""], ["P", triple.person2]],
                [["Q_GIVE", surface], ["O", triple.object], ["SOURCE_P", ""], ["TARGET_P", ""]]]
        
        return [sentence, graph_rep], answer
    
    def update_q_history(self, arg_1, arg_2, answer, supp_idxs, q_sub_type: str = None):
        q = QuestionEvent(kind=QuestionType(self.kind),
                     timestep=self.world.timestep,
                     source=arg_1,
                     ternary=arg_2,
                     target=[answer],
                     supporting_facts=supp_idxs,
                     sub_kind=q_sub_type)
        self.world.q_history[self.world.timestep] = q

    def ask_specific(self, person1, person2, object, question):
        """
        Asks a question about specific entities.
        :param question: one of the 5 possible giving questions: "gave_to", "gave" "received", "who_give", "what_give"
        """
        sentence = ""
        answer = None
        triple = None
        
        
        if question == "gave_to":
            triple = self.known_receivers[object, person2]
            t1 = (triple.object.name, triple.person2.name, triple.person1.name)
            sentence = "Who gave the {} to {}?\t{}".format(*t1)
            answer = triple.person1.name
            triple_names = t1
        elif question == "gave":
            triple = self.known_objects[object]
            t2 = (triple.object.name, None, triple.person1.name)
            sentence = "Who gave the {}?\t{}".format(triple.object.name, triple.person1.name)
            answer = triple.person1.name
            triple_names = t2
        elif question == "received":
            triple = self.known_objects[object]
            t3 = (triple.object.name, None,  triple.person2.name)
            sentence = "Who received the {}?\t{}".format(triple.object.name, triple.person2.name)
            answer = triple.person2.name
            triple_names = t3
        elif question == "who_give":
            triple = self.known_passers[person1, object]
            t4 = (triple.person1.name, triple.object.name, triple.person2.name)
            sentence = "Who did {} give the {} to?\t{}".format(*t4)
            answer = triple.person2.name
            triple_names = t4
        elif question == "what_give":
            triple = self.known_passings[person1, person2]
            t5 = (triple.person1.name, triple.person2.name, triple.object.name)
            sentence = "What did {} give to {}?\t{}".format(*t5)
            answer = triple.object.name
            triple_names = t5
        
        supp_idxs = GivingQuestion.supporting_facts_for_target(triple.object)
        supp_facts_idxs = supp_facts_str(supp_idxs)
        sentence = sentence + supp_facts_idxs
        
        self.update_q_history(*triple_names, supp_idxs=supp_idxs, q_sub_type=question)
        surface = question
        graph_rep = [[["SOURCE_P", ""], ["P", triple.person1]],
                [["TARGET_P", ""], ["P", triple.person2]],
                [["Q_GIVE", surface], ["O", triple.object], ["SOURCE_P", ""], ["TARGET_P", ""]]]
        return [sentence, graph_rep], answer
