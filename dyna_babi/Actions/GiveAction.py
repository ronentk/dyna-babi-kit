from .Action import Action
from collections import namedtuple
from ..helpers.utils import choice_np
from ..helpers.event import Event

GiveTriple = namedtuple('GiveTriple', 'kind person1 object person2')

class GiveAction(Action):
    """
    Moves an object from one person to another
    """
    def __init__(self, world):
        super().__init__(world, kind="give")
        self.locations = set()
        self.persons = set()
        self.objects = set()
        self.template = "{} {} the {} to {}."
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
        True iff some person is holding some object, and is in the same location as some other person
        :param persons: If not None, only consider these persons
        """
        if persons is None:
            persons = self.persons
        for person1 in persons:
            for object in self.objects:
                if object.holder == person1:
                    for person2 in self.persons:
                        if person2 != person1 and person2.holder == person1.holder:
                            return True
        return False
    
    
    def valid_actions(self, persons=None):
        actions = []
        if persons is None:
            persons = self.persons
        for person1 in persons:
            for object in self.objects:
                if object.holder == person1:
                    for person2 in self.persons:
                        if person2 != person1 and person2.holder == person1.holder:
                            actions.append((self.kind, person1.name, person2.name, object.name))
        return actions

    def action_sentence(self, give, person1, object, person2):
        """
        Format person1, object and person2 into self.template to create a bAbI sentence
        """
        sentence = self.template.format(person1.name, give, object.name, person2.name)
        return sentence

    def action_graph_rep(self, give, person1, object, person2):
        """
        Format person1, object and person2 into a format convertible to a graph_rep sub-structure
        """
        return [[["GIVE_SOURCE", ""], ["P", person1.name]], [["GIVE_TARGET", ""], ["P", person2.name]], [["A_GIVE", give], ["GIVE_SOURCE", ""], ["GIVE_TARGET", ""], ["O", object.name]]]

    def match_locations(self, person1, person2, override=False):
        where_person_question = self.world.get_question_by_kind("where_person")
        is_location1_known = person1 in where_person_question.known_items
        is_location2_known = person2 in where_person_question.known_items

        if is_location1_known and not is_location2_known:
            if override:
                person2.move(person1.holder)
            self.world.add_known_item(person2, person2.holder, 
                                      match_location=person1)
            self.world.update_all_neg(person2, [person2.holder])
        elif is_location2_known and not is_location1_known:
            if override:
                person1.move(person2.holder)
            self.world.add_known_item(person1, person1.holder, 
                                      match_location=person2)
            self.world.update_all_neg(person1, [person1.holder])

    def act(self, persons=None, coref: bool = False):
        """
        Performs a give action with uniform distribution over the persons and objects in the game.
        :param persons: If not None, only consider these persons
        """
        if persons is None:
            persons = self.persons
        triples = []
        for person1 in persons:
            for object in self.objects:
                if object.holder == person1:
                    for person2 in self.persons:
                        if person2 != person1 and person2.holder == person1.holder:
                            triples.append((person1, object, person2))
        triple = choice_np(triples)
        self.update_histories(triple[0], triple[1], triple[2])
        triple[0].give(triple[1], triple[2])
        self.match_locations(triple[0], triple[2])
        self.world.add_known_item(GiveTriple('give_triple', triple[0], triple[1], triple[2]), triple[0])
        self.world.add_known_item(triple[1], triple[2])
        self.world.update_all_neg_poss(triple[1])
        
        self.world.remove_known_item(triple[1], triple[0])
        give = choice_np(self.world.params.give)
        return [(self.action_sentence(give, triple[0], triple[1], triple[2]), self.action_graph_rep(give, triple[0], triple[1], triple[2]))]

    def update_histories(self, source, obj, target, coref: bool = False,
                         is_all_act: bool = False):
        source_loc_support = source.last_known_loc_support
        target_loc_support = target.last_known_loc_support
        updated_last_known_loc = False
        if not source_loc_support:
            # add supporting facts of source location to the current event
            support = target_loc_support + [self.world.timestep]
            updated_last_known_loc = True
        elif not target_loc_support:
            # add supporting facts of target location to the current event
            support = source_loc_support + [self.world.timestep]
            updated_last_known_loc = True
        else:
            # both supported, take latest
            if source_loc_support[-1] > target_loc_support[-1]:
                support = source_loc_support + [self.world.timestep]
            else:
                support = target_loc_support + [self.world.timestep] 
        location = source.holder
        ev = Event(kind=self.kind, timestep=self.world.timestep,
                                  loc_event_idxs=support, source=source.name,
                                  location=location.name,
                                  target=target.name, is_coref=coref, ternary=obj.name,
                                  updated_last_known_loc=updated_last_known_loc,
                                  is_all_act=is_all_act
                                  )
        
        if is_all_act:
            if not self.world.timestep in self.world.history:
                # part of give all - make ternary into list
                ev.ternary = [obj.name]
                self.world.history[ev.timestep] = ev
            else:
                # already added at least one give action
                self.world.history[ev.timestep].ternary.append(obj.name) 
        else:
            self.world.history[ev.timestep] = ev
            
        
        # TODO need this?
        # if either source or target or in unknown loc, but other is, the give event
        # should be used to support inference that source and target are in same loc
        source_update_last_known_loc = (source.indef_loc) and (not target.indef_loc)
        source.update_event(ev, source_update_last_known_loc)
        target_update_last_known_loc = (target.indef_loc) and (not source.indef_loc)
        target.update_event(ev, target_update_last_known_loc)
        
        obj.update_event(ev, update_last_known_change_pos=True)


    def act_specific(self, person1, person2, object, override=False,
                     is_all_act: bool = False):
        """
        Performs a specific give action. Isn't currently use for story generation, only for testing.
        :param override: if True, performs the action even if is usually illegal by the game's rules.
        """
        if (not person1.holder == person2.holder) or (not object.holder == person1):
            if not override:
                raise ValueError

        triple = [person1, object, person2]
        
        self.update_histories(triple[0], triple[1], triple[2], is_all_act=is_all_act)
        triple[0].give(triple[1], triple[2])
        self.match_locations(person1, person2, override)
        self.world.add_known_item(GiveTriple('give_triple', triple[0], triple[1], triple[2]), triple[0])
        self.world.add_known_item(triple[1], triple[2])
        self.world.remove_known_item(triple[1], triple[0])
        give = choice_np(self.world.params.give)
        return [(self.action_sentence(give, triple[0], triple[1], triple[2]), self.action_graph_rep(give, triple[0], triple[1], triple[2]))]
