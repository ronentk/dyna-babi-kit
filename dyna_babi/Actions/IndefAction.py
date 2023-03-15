from .Action import Action
from collections import namedtuple
import numpy.random as random
from ..helpers.utils import choice_np
from ..helpers.event import Event, BeliefType

IndefLocation = namedtuple('IndefLocation', 'kind location1 location2')

class IndefAction(Action):
    """
    Moves a person to some location, doesn't give full information as to which location.
    Instead, gives two options for a location.
    """
    def __init__(self, world):
        super().__init__(world, kind="indef")
        self.locations = set()
        self.persons = set()
        self.template = "{} is either in the {} or the {}."
        self.distribution = self.world.params.negate_distribution
        self.initialize()

    def initialize(self):
        entities = self.world.entities
        for entity in entities:
            if entity.kind == "location":
                self.locations.add(entity)
            if entity.kind == "person":
                self.persons.add(entity)

    def is_valid(self, persons=None, locations=None):
        """
        True iff at least one person have at least two locations to move into
        :param persons: If not None, only consider these persons
        :param locations:  If not None, only consider these locations
        """
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        for person in persons:
            for location1 in locations:
                for location2 in locations:
                    if person.holder != location1 and person.holder != location2 and location1 != location2:
                        return True
        return False

    def action_sentence(self, person, location1, location2):
        """
        Format person, location1 and location2 into self.template to create a bAbI sentence
        """
        sentence = self.template.format(person.name, location1.name, location2.name)
        return sentence

    def action_graph_rep(self, person, location1, location2):
        """
        Format person and location into a format convertible to a graph_rep sub-structure
        """
        return [[["A_INDEF", ""], ["P", person.name], ["L", location1.name], ["L", location2.name]]]

    def act(self, persons=None, locations=None, coref: bool = False):
        """
        Performs an indefinite action with uniform distribution over the persons and locations in the game.
        :param persons: If not None, only consider these persons
        :param locations:  If not None, only consider these locations
        """
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        triples = []
        for person in persons:
            for location1 in locations:
                for location2 in locations:
                    if person.holder != location1 and person.holder != location2 and location1 != location2:
                        triples.append((person, location1, location2))
        triple = choice_np(triples)
        person, location1, location2 = triple
        location = random.choice([location1, location2])
        last_location = person.holder
        person.move(location)
        self.update_histories(person, location1, location2, coref, gold_belief=BeliefType.INDEF)
        self.world.remove_known_item(person, last_location)
        self.world.add_known_item(person, IndefLocation('indef_location', location1, location2))
        self.world.update_all_neg(person, [location1, location2])
        return [(self.action_sentence(*triple), self.action_graph_rep(*triple))]

    def update_histories(self, person, location1, location2, coref: bool = False, 
                         gold_belief: BeliefType = BeliefType.KNOWN):
        ev = Event(kind=self.kind, timestep=self.world.timestep,
                                  source=person.name,
                                  location=person.holder.name,
                                  target=[location1.name, location2.name], is_coref=coref,
                                  gold_belief=gold_belief)
        self.world.history[ev.timestep] = ev
        person.update_event(ev)

            
    def act_specific(self, person, location1, location2, override=False, coref=False):
        """
        Performs a specific indefinite action. Isn't currently use for story generation, only for testing.
        :param override: if True, performs the action even if is usually illegal by the game's rules.
        """
        if not(person.holder != location1 and person.holder != location2 and location1 != location2):
            if not override:
                raise ValueError
        location = random.choice([location1, location2])
        last_location = person.holder
        person.move(location)
        self.update_histories(person, location1, location2, coref, gold_belief=BeliefType.INDEF)
        self.world.remove_known_item(person, last_location)
        self.world.add_known_item(person, IndefLocation('indef_location', location1, location2))
        
        return [(self.action_sentence(person, location1, location2), self.action_graph_rep(person, location1, location2))]
