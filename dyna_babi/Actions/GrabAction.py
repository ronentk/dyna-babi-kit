from .Action import Action
import numpy as np
import numpy.random as random
from ..helpers.utils import choice_np, sorted_item_set
from ..helpers.event import Event

class GrabAction(Action):
    def __init__(self, world):
        super().__init__(world, kind="grab")
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
        True is some person is in the same location as some object
        :param persons: If not None, only consider these persons
        """
        if persons is None:
            persons = self.persons
        for person in persons:
            for object in self.objects:
                if person.holder == object.holder:
                    return True
        return False
    
    def valid_actions(self, persons=None):
        actions = []
        if persons is None:
            persons = self.persons
        for person in persons:
            for object in self.objects:
                if person.holder == object.holder:
                    actions.append((self.kind, person.name, object.name))
        return actions

    def action_sentence(self, grab, person, object):
        """
        Format person and object into self.template to create a bAbI sentence
        """
        postfix = " there" if random.uniform(0, 1) < 0 else "" # should not happen (see #15)
        sentence = self.template.format(person.name, grab, object.name, postfix)
        return sentence

    def action_graph_rep(self, grab, person, object):
        """
        Format person and object into a format convertible to a graph_rep sub-structure
        """
        return [[["A_GRAB", grab], ["P", person.name], ["O", object.name]]]

    def match_locations(self, person, object, override=False):
        where_person_question = self.world.get_question_by_kind("where_person")
        where_object_question = self.world.get_question_by_kind("where_object")
        is_person_known = person in where_person_question.known_items
        is_object_known = object in where_object_question.known_items

        if is_object_known and not is_person_known:
            if override:
                person.move(object.holder)
            self.world.add_known_item(person, person.holder, match_location=object)
            self.world.update_all_neg(person, [person.holder])


    def act(self, persons=None, coref: bool = False):
        """
        Performs a grab action with uniform distribution over the persons and objects in the game.
        :param persons: If not None, only consider these persons
        """
        if persons is None:
            persons = self.persons
        pairs = []
        for person in sorted_item_set(persons):
            for object in sorted_item_set(self.objects):
                if person.holder == object.holder:
                    pairs.append((person, object))
        pair = choice_np(pairs)
        obj = pair[1]
        person = pair[0]
        names = [(x[0].name, x[1].name) for x in pairs]
        self.update_histories(person, obj, coref)
        self.match_locations(pair[0], pair[1])
        pair[0].grab(pair[1])
        self.world.add_known_item(pair[1], pair[0])
        self.world.update_all_neg_poss(pair[1])

        grab = choice_np(self.world.params.grab)
        return [(self.action_sentence(grab, pair[0], pair[1]), self.action_graph_rep(grab, pair[0], pair[1]))]
    
    def update_histories(self, person, obj, coref: bool = False):
        location = person.holder
        ev = Event(kind=self.kind, timestep=self.world.timestep,
                                  loc_event_idxs=person.last_known_loc_support,
                                  location=location.name,
                                  source=person.name,
                                  target=obj.name, is_coref=coref)
        self.world.history[ev.timestep] = ev
        person.update_event(ev)
        obj.update_event(ev, update_last_known_change_pos=True)

    def act_specific(self, person, object, override=False, coref: bool = False):
        """
        Performs a specific grab action. Isn't currently use for story generation, only for testing.
        :param override: if True, performs the action even if is usually illegal by the game's rules.
        """
        if person.holder != object.holder:
            if not override:
                raise ValueError
        self.update_histories(person, object, coref=coref)
        self.match_locations(person, object, override)
        person.grab(object)
        
        self.world.add_known_item(object, person)
        

        grab = choice_np(self.world.params.grab)
        return [(self.action_sentence(grab, person, object), self.action_graph_rep(grab, person, object))]
