import numpy as np
import numpy.random as random
from . import WherePersonQuestion, WhereObjectQuestion, WhereWasObjectQuestion, GivingQuestion, YesNoQuestion, CountingQuestion, ListQuestion


class QuestionList(object):
    """
    Manages the list of legal questions in a bAbI game.
    """
    def __init__(self, world, questions=None, distribution=None):
        """
        :param world: A World object
        :param questions: a list of legal questions
        :param distribution: the distribution from which to choose an questions when requested
        """
        if not questions:
            questions = []
        if not distribution:
            distribution = []
        self.world = world
        self.questions = questions
        self.distribution = distribution

    def init_from_params(self, questions, distribution):
        """
            Creates Question objects for the question names in "questions" and adds them to self.questions
            :param questions: a list of legal question names (strings)
            :param distribution: the distribution from which to choose an questions when requested
            """
        self.distribution = distribution
        for question in questions:
            if question == "where_person":
                self.questions.append(WherePersonQuestion.WherePersonQuestion(world=self.world))
            if question == "where_object":
                self.questions.append(WhereObjectQuestion.WhereObjectQuestion(world=self.world))
            if question == "where_was_object":
                self.questions.append(WhereWasObjectQuestion.WhereWasObjectQuestion(world=self.world))
            if question == "giving":
                self.questions.append(GivingQuestion.GivingQuestion(world=self.world))
            if question == "yes_no":
                self.questions.append(YesNoQuestion.YesNoQuestion([entity for entity in self.world.entities if entity.kind == 'location'],
                                                                  world=self.world))
            if question == "counting":
                self.questions.append(CountingQuestion.CountingQuestion([entity for entity in self.world.entities if entity.kind == 'person'], world=self.world))
            if question == "list":
                self.questions.append(ListQuestion.ListQuestion([entity for entity in self.world.entities if entity.kind == 'person'], world=self.world))

        if not [question for question in self.questions if question.kind == "where_person"]:
            self.questions.append(WherePersonQuestion.WherePersonQuestion(world=self.world))
            self.distribution.append(0.0)
        if not [question for question in self.questions if question.kind == "where_object"]:
            self.questions.append(WhereObjectQuestion.WhereObjectQuestion(world=self.world))
            self.distribution.append(0.0)

    def add_known_item(self, a, b, match_location=None):
        """
        Adds new information about an entity, that should be known the reader of a bAbI story (if the reader reads well)
        Usually a will be some entity and b will be the entity that is known to hold a (but not necessarily).
        """
        for question in self.questions:
            question.add_known_item(a, b, t=self.world.timestep,
                                    match_location=match_location)

    def remove_known_item(self, a, b):
        """
        Adds new information about an entity, that should be known the reader of a bAbI story (if the reader reads well)
        Usually a will be some entity and b will be the entity that is known to  no longer hold a (but not necessarily).
        """
        for question in self.questions:
            question.remove_known_item(a, b)

    def forget(self):
        """
        Deletes all previous information about what is holding what.
        """
        for question in self.questions:
            question.forget()

    def can_ask(self):
        """
        Can any question be asked. A question can only be asked if there is some information known to the reader that
        the question can be asked about.
        """
        for idx in range(len(self.questions)):
            if self.questions[idx].is_valid() and self.distribution[idx] != 0:
                return True
        return False

    def ask_question(self):
        """
        Chooses a question from self.questions according to self.distribution.
        """
        valid_questions_idx = [index for index, question in enumerate(self.questions) if question.is_valid()]
        questions = [self.questions[idx] for idx in valid_questions_idx]
        weights = np.array([self.distribution[idx] for idx in valid_questions_idx])
        p = weights / weights.sum()
        question = random.choice(questions, p=p)
        
        return question.ask()
    
    def ask_all_valid_questions(self):
        all_qs = []
        for q in sorted(self.questions, key=lambda x: x.kind):
            # check that non-zero prob to ask
            if self.distribution[self.questions.index(q)] > 0.0:
                all_qs += q.all_valid_questions()
        return all_qs
