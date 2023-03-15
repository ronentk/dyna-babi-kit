import numpy.random as random
from enum import Enum


class QuestionType(str, Enum):
    COUNT = "counting"
    YES_NO = "yes_no"
    WHERE_OBJ = "where_object"
    WHERE_PERSON = "where_person"
    GIVING = "giving"
    WHERE_WAS_OBJ = "where_was_object"
    LIST = "list"
    YES_NO_OBJ_LOC = "yes_no_obj_loc"
    YES_NO_PERS_OBJ = "yes_no_pers_obj"


class Question(object):
    def __init__(self, world, template="", kind="", known_items=None):
        """
        :param template: the string template for asking the question
        :param kind: A string identifying the the question
        :param known_items: The items who's information is known to the reader. If this collection is not empty,
                            the question can be asked.
        """
        self.world= world
        self.template = template
        self.kind = kind
        if known_items is None:
            self.known_items = set()
        else:
            self.known_items = known_items

    def all_valid(self):
        """ 
        Return list of all entities (or iterable tuples of entities) for which
        valid specific questions can be asked (`ask_specific`).
        """
        raise NotImplementedError()
        
    def all_valid_questions(self):
        qs = []
        for entities in self.all_valid():
            self.world.timestep += 1
            if type(entities) == tuple:
                q = self.ask_specific(*entities)
            else:
                q = self.ask_specific(entities)
            qs.append(q)
            
        return qs
        