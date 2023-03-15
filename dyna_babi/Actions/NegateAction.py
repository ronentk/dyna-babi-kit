from .Action import Action
import numpy.random as random
from ..helpers.utils import choices_np, choice_np
from ..helpers.event import Event, BeliefType

class NegateAction(Action):
    """
    Moves a person to some location, but may describe it as a negative statement ("person is no longer in location")
    """
    def __init__(self, world):
        super().__init__(world, kind="negate")
        self.locations = set()
        self.persons = set()
        self.template = "{} is{} in the {}."
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
        True iff at least one person have at least one location to move into
        :param persons: If not None, only consider these persons
        :param locations:  If not None, only consider these locations
        """
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        for person in persons:
            for location in locations:
                if person.holder != location:
                    return True
        return False

    def action_sentence(self, is_or_not_alias, person, location):
        """
        Format person, is_or_not and location into self.template to create a bAbI sentence
        """
        sentence = self.template.format(person.name, is_or_not_alias, location.name)
        return sentence

    def action_graph_rep(self, is_or_not_alias, person, location):
        """
        Format person and location into a format convertible to a graph_rep sub-structure
        """
        return [[["A_NEGATE", is_or_not_alias], ["P", person.name], ["L", location.name]]]

    def act(self, persons=None, locations=None, coref: bool = False):
        """
        Performs a negate action with uniform distribution over the persons and locations in the game.
        :param persons: If not None, only consider these persons
        :param locations:  If not None, only consider these locations
        """
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        person = choice_np(persons)
        last_location = person.holder
        location = person.holder
        while location == person.holder:
            location = choice_np(locations)
        person.move(location)
        # is_or_not determines if it's a regular sentence or a negation sentence. if negative (False) it's a negation sentence
        is_or_not = choices_np([False, True], self.world.params.negate_distribution)
        is_or_not_alias = "" if is_or_not else " " + choice_np(self.world.params.negate)
        if is_or_not:
            sentence = self.action_sentence(is_or_not_alias, person, location)
            graph_rep = self.action_graph_rep("is", person, location)
            self.update_histories(person, location, coref)
            self.world.add_known_item(person, location)
            self.world.update_all_neg(person, [location])
            
        else:
            sentence = self.action_sentence(is_or_not_alias, person, last_location)
            graph_rep = self.action_graph_rep(is_or_not_alias, person, last_location)
            self.update_histories(person, last_location, coref, gold_belief=BeliefType.NEGATED)
            self.world.remove_known_item(person, last_location)
            self.world.update_all_neg(person, [last_location], belief_prob=-1.0) # set all to unknown
            
        return [(sentence, graph_rep)]

    def update_histories(self, person, location, coref: bool = False, 
                         gold_belief: BeliefType = BeliefType.KNOWN):
        if gold_belief == BeliefType.KNOWN:
            loc_event_idxs = [self.world.timestep]
        else:
            # no last known loc index
            loc_event_idxs = []
        update_last_known_loc = len(loc_event_idxs) > 0
        ev = Event(kind=self.kind, timestep=self.world.timestep,
                                  source=person.name,
                                  loc_event_idxs=loc_event_idxs,
                                  target=location.name, is_coref=coref,
                                  location=person.holder.name,
                                  gold_belief=gold_belief,
                                  ternary=person.holder.name,
                                  updated_last_known_loc=update_last_known_loc)
        self.world.history[ev.timestep] = ev
        person.update_event(ev, update_last_known_loc=update_last_known_loc)


    def act_specific(self, person, negate, location, override=False, coref: bool = False):
        """
        Performs a specific negate action. Isn't currently use for story generation, only for testing.
        :param override: if True, performs the action even if is usually illegal by the game's rules.
        """
        if negate:
            if person.holder != location:
                if not override:
                    raise ValueError
        else:
            if person.holder == location:
                if not override:
                    raise ValueError

        if not negate:
            person.move(location)
            self.update_histories(person, location, coref)
            
            sentence = self.template.format(person.name, negate, location.name)
            graph_rep = self.action_graph_rep("is", person, location)
            self.world.add_known_item(person, location)
        else:
            new_location = location
            while new_location == location:
                new_location = choice_np(self.locations)
            person.move(new_location)
            self.update_histories(person, location, coref, gold_belief=BeliefType.NEGATED)
            sentence = self.template.format(person.name, " " + negate, location.name)
            graph_rep = self.action_graph_rep(negate, person, location)
            self.world.remove_known_item(person, location)
        return [(sentence, graph_rep)]
