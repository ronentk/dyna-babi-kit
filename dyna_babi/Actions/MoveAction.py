from .Action import Action
import numpy.random as random
from ..helpers.utils import choice_np
from ..helpers.event import Event


class MoveAction(Action):
    def __init__(self, world):
        super().__init__(world, kind="move")
        self.locations = set()
        self.persons = set()
        self.template = "{} {} to the {}."
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
    
    def valid_actions(self, persons=None, locations=None):
        actions = []
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        for person in persons:
            for location in locations:
                if person.holder != location:
                    actions.append((self.kind, person.name, location.name))
        return actions
                    
        

    def action_sentence(self, move, person, location):
        """
        Format person and location into self.template to create a bAbI sentence
        """
        sentence = self.template.format(person.name, move, location.name)
        return sentence

    def action_graph_rep(self, move, person, location):
        """
        Format person and location into a format convertible to a graph_rep sub-structure
        """
        return [[["A_MOVE", move], ["P", person.name], ["L", location.name]]]

    def act(self, persons=None, locations=None, coref: bool = False):
        """
        Performs a move action with uniform distribution over the persons and locations in the game.
        :param persons: If not None, only consider these persons
        :param locations:  If not None, only consider these locations
        """
        if persons is None:
            persons = self.persons
        if locations is None:
            locations = self.locations
        
        person_list = list(persons)
        person = choice_np(person_list)
        location = person.holder
        old_location = location
        while location == person.holder:
            location = choice_np(locations)  
        person.move(location)
        
        self.update_histories(person, location, coref)
        
        self.world.add_known_item(person, location, kind=self.kind)

        # record that person isn't at any other location
        self.world.update_all_neg(person, [location])
        
        for object in person.holds:
            self.world.add_known_item(object, location, kind=self.kind)

        
        moves = self.world.params.move
        move = choice_np(moves)
        return [(self.action_sentence(move, person, location), self.action_graph_rep(move, person, location))]

    def update_histories(self, person, location, coref: bool = False):
        ev = Event(kind=self.kind, timestep=self.world.timestep,
                                  source=person.name,
                                  location=location.name,
                                  loc_event_idxs=[self.world.timestep],
                                  target=location.name, is_coref=coref,
                                  updated_last_known_loc=True)
        
        if ev.timestep in self.world.history:
            # conjunction - make source into list
            parallel_ev = self.world.history[ev.timestep]
            self.world.history[ev.timestep].source = [ev.source, parallel_ev.source]
            self.world.history[ev.timestep].is_conj = True
        else:
            self.world.history[ev.timestep] = ev
        
        person.update_event(ev, update_last_known_loc=True)

        

    def act_specific(self, person, location, override=False, coref: bool = False):
        """
        Performs a specific move action. Isn't currently use for story generation, only for testing.
        :param override: if True, performs the action even if is usually illegal by the game's rules.
        """
        game_location = person.holder
        if game_location == location:
            if not override:
                raise ValueError
        person.move(location)
        self.update_histories(person, location, coref=coref)
        
        self.world.add_known_item(person, location)

        for object in person.holds:
            self.world.add_known_item(object, location)

        
        
        move = choice_np(self.world.params.move)
        return [(self.action_sentence(move, person, location), self.action_graph_rep(move, person, location))]
