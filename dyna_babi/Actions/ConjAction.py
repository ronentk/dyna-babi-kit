from .Action import Action
from .MoveAction import MoveAction
from ..helpers.utils import choice_np
from ..helpers.event import Event

class ConjAction(Action):
    """
    Moves two persons to the same location. Describes it in one sentence.
    """
    def __init__(self, world):
        super().__init__(world, kind="conj")
        self.locations = set()
        self.persons = set()
        self.objects = set()
        self.template = "{} and {} {} to the {}."
        self.base_action = MoveAction(self.world)
        self.initialize()

    def initialize(self):
        entities = self.world.entities
        for entity in entities:
            if entity.kind == "location":
                self.locations.add(entity)
            if entity.kind == "person":
                self.persons.add(entity)
            if entity.kind == "object":
                self.objects.add(entity)

    def is_valid(self, persons=None, locations=None):
        """
        True iff at least two persons can move into the same location
        :param persons: If not None, only consider these persons
        :param locations:  If not None, only consider these locations
        """
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        for person1 in persons:
            for person2 in persons:
                for location in locations:
                    if person1 != person2 and self.base_action.is_valid([person1], [location]) and self.base_action.is_valid([person2], [location]):
                        return True
        return False

    def action_sentence(self, move, person1, person2, location):
        """
        Format person1, person2 and location into self.template to create a bAbI sentence
        """
        sentence = self.template.format(person1.name, person2.name, move, location.name)
        return sentence

    def action_graph_rep(self, move, person1, person2, location):
        """
        Format person1, person2 and location into a format convertible to a graph_rep sub-structure
        """
        return [[["A_MOVE", ""], ["P", person1.name], ["P", person2.name], ["L", location.name]]]

    def act(self, persons=None, locations=None, coref: bool = False):
        """
        Performs a conjunction action with uniform distribution over the persons and locations in the game.
        :param persons: If not None, only consider these persons
        :param locations:  If not None, only consider these locations
        """
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        triples = []
        for person1 in persons:
            for person2 in persons:
                for location in locations:
                    if person1 != person2 and self.base_action.is_valid([person1], [location]) and self.base_action.is_valid([person2], [location]):
                        triples.append((person1, person2, location))

        triple = choice_np(triples)
        move = choice_np(self.world.params.move)
        quadruple = [move] + list(triple)

        self.base_action.act([triple[0]], [triple[2]], coref=coref)
        self.base_action.act([triple[1]], [triple[2]], coref=coref)
        return [(self.action_sentence(*quadruple), self.action_graph_rep(*quadruple))]

    def act_specific(self, person1, person2, location, override=False, coref: bool = False):
        """
        Performs a specific conjunction action. Isn't currently use for story generation, only for testing.
        :param override: if True, performs the action even if is usually illegal by the game's rules.
        """
        if (not self.base_action.is_valid([person1], [location])) or (not self.base_action.is_valid([person2], [location])):
            if not override:
                raise ValueError

        self.base_action.act_specific(person1, location, override, coref=coref)
        self.base_action.act_specific(person2, location, override, coref=coref)

        triple = (person1, person2, location)
        move = choice_np(self.world.params.move)
        quadruple = [move] + list(triple)

        return [(self.action_sentence(*quadruple), self.action_graph_rep(*quadruple))]
