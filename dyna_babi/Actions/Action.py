import numpy.random as random
from enum import Enum


class ActionType(str, Enum):
    MOVE = "move"
    GRAB = "grab"
    DROP = "drop"
    GIVE = "give"
    COREF = "coref"
    CONJ = "conj"
    COMPOUND = "compound"
    NEGATE = "negate"
    INDEF = "indef"
    GIVE_ALL = "give_all" # experimental
    DROP_ALL = "drop_all" # experimental

class Action(object):
    def __init__(self, world, kind):
        """
        :param world: A World object
        :param kind: A string identifying the the action
        """
        self.kind = kind
        self.world = world
        
        
    def valid_actions(self):
        return []