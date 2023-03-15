import numpy.random as random
import numpy as np
from . import MoveAction, GrabAction, DropAction, GiveAction, CorefAction, ConjAction, CompoundAction, NegateAction, IndefAction





class ActionList(object):
    """
    Manages the list of legal actions in a bAbI game.
    """
    def __init__(self, world, actions=None, distribution=None):
        """
        :param world: A World object
        :param actions: a list of legal actions
        :param distribution: the distribution from which to choose an actions when requested
        """
        if not actions:
            actions = []
        if not distribution:
            distribution = []
        self.world = world
        self.actions = actions
        self.distribution = distribution

    def init_from_params(self, actions, distribution):
        """
        Creates Action objects for the action names in "actions" and adds them to self.actions
        :param actions: a list of legal action names (strings)
        :param distribution: the distribution from which to choose an actions when requested
        """
        self.distribution = distribution
        for action in actions:
            if action == "move":
                self.actions.append(MoveAction.MoveAction(self.world))
            if action == "grab":
                self.actions.append(GrabAction.GrabAction(self.world))
            if action == "drop":
                self.actions.append(DropAction.DropAction(self.world))
            if action == "give":
                self.actions.append(GiveAction.GiveAction(self.world))
            if action == "coref":
                self.actions.append(CorefAction.CorefAction(self.world))
            if action == "conj":
                self.actions.append(ConjAction.ConjAction(self.world))
            if action == "compound":
                self.actions.append(CompoundAction.CompoundAction(self.world))
            if action == "negate":
                self.actions.append(NegateAction.NegateAction(self.world))
            if action == "indef":
                self.actions.append(IndefAction.IndefAction(self.world))

    def can_act(self, persons=None):
        """
        Can any action be performed. Note that in the usual case we have at least one person and two locations,
        and move actions are allowed - in that case some action can always be performed.
        :param persons: If "persons" is not None, only consider actions that involves the persons in "persons"
        """
        for idx in range(len(self.actions)):
            if self.actions[idx].is_valid(persons) and self.distribution[idx] != 0.0:
                return True
        raise ActionError("No action is available with probability greater than 0. Reconfigure parameter file.")

    def make_action(self, persons=None, coref: bool = False):
        """
        Chooses an action from self.actions according to self.distribution.
        :param persons: If "persons" is not None, only consider actions that involves the persons in "persons"
        """
        self.world.timestep += 1
        valid_actions_idx = [index for index, action in enumerate(self.actions) if action.is_valid(persons)]
        actions = [self.actions[idx] for idx in valid_actions_idx]
        weights = np.array([self.distribution[idx] for idx in valid_actions_idx])
        p = weights / weights.sum()
        action = random.choice(actions, p=p)
        return action.act(persons, coref=coref)


class ActionError(Exception):
    pass
