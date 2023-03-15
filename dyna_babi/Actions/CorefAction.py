from .Action import Action
from . import ActionList
from ..helpers.utils import choice_np
from ..helpers.event import Event

class CorefAction(Action):
    """
    Acts twice in a row, on the same person both times. The second time, the person is only described by his or her
    pronoun (coreference). Actions must be from self.action_list
    """
    def __init__(self, world):
        super().__init__(world, kind="coref")
        self.locations = set()
        self.persons = set()
        self.objects = set()
        self.action_list = ActionList.ActionList(self.world)
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

        self.action_list.init_from_params(self.world.params.coref, self.world.params.coref_distribution)

    def is_valid(self, persons=None):
        """
        True iff at least one action from self.action_list is valid
        :param persons: If not None, only consider these persons
        """
        return self.action_list.can_act(persons)

    def action_dbca(self, old_dbca, alias):
        """
        Create new and correct coreference dbca element out of the old dbca element that did not reflect coreference
        """
        old_component = old_dbca[0]
        component1 = [["CO-REF", alias]]
        component2 = [old_component[0]]

        component1 = component1 + [[subcomponent[0], subcomponent[1]] for subcomponent in old_component[1:] if subcomponent[0] == "P"]
        component2 = component2 + [["CO-REF", alias]] + [[subcomponent[0], subcomponent[1]] for subcomponent in old_component[1:] if subcomponent[0] != "P"]
        return [component1, component2]

    def act(self, persons=None, coref: bool = False):
        """
        Performs a coreference action with uniform distribution over the persons in the game.
        :param persons: If not None, only consider these persons
        """
        self.world.timestep -= 1 # since we will call make action again
        if persons is None:
            persons = self.persons
        sentences = []
        sentences.extend(self.action_list.make_action())
        person_name = sentences[0][0].split()[0]
        persons = [person for person in persons if person.name == person_name]
        coref_sentence = list(self.action_list.make_action(persons, coref=True)[0])
        alias = self.world.params.entity_coreference_map[person_name][0]
        prefix = choice_np(self.world.params.coreference_prefixes)
        coref_sentence[0] = prefix + " " + coref_sentence[0].replace(person_name, alias)
        coref_sentence[1] = self.action_dbca(coref_sentence[1], alias)
        sentences.append(coref_sentence)
        return sentences
