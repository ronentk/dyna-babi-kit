from .Action import Action
from .ConjAction import ConjAction
from ..helpers.utils import choice_np

class CompoundAction(Action):
    """
    Moves two persons to the some place twice in a row. The second time, describes them as "they" (coreference).
    """
    def __init__(self, world):
        super().__init__(world, kind="compound")
        self.locations = set()
        self.persons = set()
        self.objects = set()
        self.base_action = ConjAction(self.world)
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
        True iff at base_action is valid
        :param persons: If not None, only consider these persons
        """
        return self.base_action.is_valid(persons)

    def action_graph_rep(self, old_graph_rep, alias):
        """
        Create new and correct coreference graph_rep element out of the old graph_rep element that did not reflect coreference
        """
        old_component = old_graph_rep[0]
        component1 = [["CO-REF", alias]]
        component2 = [old_component[0]]

        component1 = component1 + [[subcomponent[0], subcomponent[1]] for subcomponent in old_component[1:] if subcomponent[0] == "P"]
        component2 = component2 + [["CO-REF", alias]] + [[subcomponent[0], subcomponent[1]] for subcomponent in old_component[1:] if subcomponent[0] != "P"]
        return [component1, component2]

    def act(self, persons=None, coref: bool = False):
        """
        Performs a compound action with uniform distribution over the persons and locations in the game.
        :param persons: If not None, only consider these persons
        """
        if persons is None:
            persons = self.persons
        sentences = []
        sentences.extend(self.base_action.act())
        self.world.timestep += 1 # since we will call make action again
        components = sentences[0][0].split()
        person1_name, person2_name = components[0], components[2]
        persons = [person for person in persons if (person.name == person1_name or person.name == person2_name)]
        coref_sentence = list(self.base_action.act(persons, coref=True)[0])
        alias = "they"
        prefix = choice_np(self.world.params.coreference_prefixes)
        coref_sentence[0] = prefix + " " + coref_sentence[0].replace(person1_name + " and " + person2_name, alias).replace(person2_name + " and " + person1_name, alias)
        coref_sentence[1] = self.action_graph_rep(coref_sentence[1], alias)
        sentences.append(coref_sentence)
        return sentences
