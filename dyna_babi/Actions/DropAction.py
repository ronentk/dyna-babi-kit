from .Action import Action
import numpy.random as random
from ..helpers.utils import choice_np
from ..helpers.event import Event

class DropAction(Action):
    def __init__(self, world):
        super().__init__(world, kind="drop")
        self.locations = set()
        self.persons = set()
        self.objects = set()
        self.template = "{} {} the {}{}."
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

    def is_valid(self, persons=None):
        """
        True iff some person holds some object
        :param persons: If not None, only consider these persons
        """
        if persons is None:
            persons = self.persons
        for person in persons:
            for object in self.objects:
                if object.holder == person:
                    return True
        return False
    
    def valid_actions(self, persons=None):
        actions = []
        if persons is None:
            persons = self.persons
        for person in persons:
            for object in self.objects:
                if object.holder == person:
                    actions.append((self.kind, person.name, object.name))
        return actions

    def action_sentence(self, drop, person, object):
        """
        Format person and object into self.template to create a bAbI sentence
        """
        postfix = " there" if random.uniform(0, 1) < 0 else "" # should not happen (see #15)
        sentence = self.template.format(person.name, drop, object.name, postfix)
        return sentence

    def action_graph_rep(self, drop, person, object):
        """
        Format person and object into a format convertible to a graph_rep sub-structure
        """
        return [[["A_DROP", drop], ["P", person.name], ["O", object.name]]]

    def act(self, persons=None, coref: bool = False, is_all_action: bool = False):
        """
        Performs a drop action with uniform distribution over the persons and objects in the game.
        :param persons: If not None, only consider these persons
        """
        if persons is None:
            persons = self.persons
        pairs = []
        for person in persons:
            for object in self.objects:
                if object.holder == person:
                    pairs.append((object.holder, object))
        pair = choice_np(pairs)
        pair[0].drop(pair[1])
        obj = pair[1]
        person = pair[0]
        self.update_histories(person, obj, coref, is_all_action)
        self.world.remove_known_item(pair[1], pair[0])
        
        drop = choice_np(self.world.params.drop)
        return [(self.action_sentence(drop, pair[0], pair[1]), self.action_graph_rep(drop, pair[0], pair[1]))]
    
    def update_histories(self, person, obj, coref: bool = False, is_all_action: bool = False):
        location = person.holder
        ev = Event(kind=self.kind, timestep=self.world.timestep,
                                   location=location.name,
                                  loc_event_idxs=person.last_known_loc_support,
                                  source=person.name,
                                  target=obj.name, is_coref=coref,
                                  is_all_act=is_all_action)
        if is_all_action:
            if not self.world.timestep in self.world.history:
                # part of drop all - make target into list
                ev.target = [obj.name]
                self.world.history[ev.timestep] = ev
            else:
                # already added at least one drop action
                self.world.history[ev.timestep].target.append(obj.name) 
        else:
            self.world.history[ev.timestep] = ev
        person.update_event(ev)
        obj.update_event(ev, update_last_known_change_pos=True)

    def act_specific(self, person, object, override=False, coref: bool = False, 
                     is_all_action: bool = False):
        """
        Performs a specific drop action. Isn't currently use for story generation, only for testing.
        :param override: if True, performs the action even if is usually illegal by the game's rules.
        """
        if object.holder != person:
            if not override:
                raise ValueError
        person.drop(object)
        self.update_histories(person, object, coref=coref, is_all_action=is_all_action)
        self.world.remove_known_item(object, person)
        drop = choice_np(self.world.params.drop)
        return [(self.action_sentence(drop, person, object), self.action_graph_rep(drop, person, object))]
