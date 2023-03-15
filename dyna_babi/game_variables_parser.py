from typing import List, Dict
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from .helpers.utils import RANDOM_SEED


@dataclass_json
@dataclass
class GameVariables:
    """
    Reresents the entities and relations in a bAbI game.
    
    :param persons: a list persons names (strings)
    :param objects: a list objects names (strings)
    :param locations: a list locations names (strings)
    :param can_grab: a relation between entities
    :param can_move_to: a relation between entities
    """


    persons: List[str] = field(default_factory=lambda: ["Daniel",                                                         "Mary",
                                                        "John","Sandra",
                                                        "Fred","Bill",
                                                        "Jeff","Julie"])
    objects: List[str] = field(default_factory=lambda: ["apple",
                                                        "football",
                                                        "milk"])
    locations: List[str] = field(default_factory=lambda: ["bathroom",
            "bedroom", "garden", "hallway", "kitchen", "office",
            "cinema", "park", "school"])
    # Relations such as can_grab and can_move_to are not use in the current implementation.
    # Instead, it is assumed tha any person can grab any object and move to any location.
    can_grab: Dict = field(default_factory=lambda: {})
    can_move_to: Dict = field(default_factory=lambda: {})

    def __add__(self, other):
        persons = set(self.persons)
        persons.update(other.persons)
        persons = list(persons)
        objects = set(self.objects)
        objects.update(other.objects)
        objects = list(objects)
        locations = set(self.locations)
        locations.update(other.locations)
        locations = list(locations)
        can_grab = dict(self.can_grab)
        can_grab.update(other.can_grab)
        can_move_to = dict(self.can_move_to)
        can_move_to.update(other.can_move_to)
        return GameVariables(persons, objects, locations, can_grab, can_move_to)



def default_question_distribution(question):
    if question == "conj" and question == "compound":
        return 0.5
    else:
        return 1


@dataclass_json
@dataclass
class StoryParameters:
    """ 
    Represents the rules and properties that determine the writing of bAbI stories in the data.
    
    :param samples: the number of samples (stories) in the new dataset
    :param n_questions: the number of questions in each story
    :param question_probability: probability of asking a question (as opposed to performing an action)
    :param actions_before_question: minimum number of actions before a question can be asked
    :param questions: the types of questions that may be asked
    :param questions_distribution: the distribution from which to choose a question type
    :param max_actions: maximal number of actions that may be performed in a story. currently no in use
    :param actions: the types of actions that may be asked
    :param actions_distribution: the distribution from which to choose an actions type
    :param coref: the actions over which the generator may use coreference
    :param coref_distribution: the distribution from which to choose an actions to perform coreference over
    :param move: synonyms of move
    :param grab: synonyms of grab
    :param drop: synonyms of drop
    :param give: synonyms of give
    :param negate: synonyms of negate
    :param negate_distribution: the distribution from which to choose whether to negate an action or not (only applies to NegateAction)
    :param coreference_prefixes: prefixes that may appear at the start of a coreference sentence
    :param entity_coreference_map: maps each entity in the game to it's appropriate pronoun for coreference
    :param seed: Random seed for re-produceability. Default is RANDOM_SEED which will randomly generate seed.
    :param exhaustive: When asking a question, generate all possible questions at the current world state.
    param extra_exh_yes_no: For yes no questions, generate all possible questions at the current world state rather than sub sample of them.
    
    """
    samples: int = 11000
    seed: int = RANDOM_SEED
    name: str = "name"
    n_questions: int = 5
    question_probability: float = 0.5
    actions_before_question: int = 2
    max_actions: int = 0
    questions: List[str] = field(default_factory=lambda: ['where_object', 'where_person', 'where_was_object', 'giving', 'yes_no', 'counting', 'list'])
    questions_distribution: List[float] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])
    actions: List[str] = field(default_factory=lambda: ['move', 'grab', 'drop', 'give', 'coref', 'conj', 'compound', 'negate', 'indef'])
    actions_distribution: List[float] = field(default_factory=lambda: [1,1,1,1,1,0.5,0.5,1,1])
    coref: List[str] = field(default_factory=lambda: ["move","grab","drop"])
    coref_distribution: List[float] = field(default_factory=lambda: [1,1,1])
    move: List[str] = field(default_factory=lambda: [
        "journeyed", "moved", "travelled", "went", "went back"
        ])
    grab: List[str] = field(default_factory=lambda: [
        "grabbed", "picked up", "took", "got"
        ])
    drop: List[str] = field(default_factory=lambda: [
        "discarded", "dropped", "left" , "put down"
        ])
    give: List[str] = field(default_factory=lambda: [
        "gave", "handed", "passed"
        ])
    negate: List[str] = field(default_factory=lambda: [
        "not", "no longer"
        ])
    negate_distribution: List[float] = field(default_factory=lambda: [0.5, 0.5])
    coreference_prefixes: List[str] = field(default_factory=lambda: [
        "After that", "Following that", "Afterwards", "Then"
        ])
    entity_coreference_map: Dict = field(default_factory=lambda: {
        "John": ["he"],
        "Daniel": ["he"],
        "Fred": ["he"],
        "Bill": ["he"],
        "Jeff": ["he"],
        "Julie": ["she"],
        "Sandra": ["she"],
        "Mary": ["she"]
        
        })
    exhaustive: bool = False
    extra_exh_yes_no: bool = False
    
    
    def __post__init(self):
        if self.questions_distribution == []:
            self.questions_distribution = [default_question_distribution(question) for question in self.questions]

        
        
    

def get_game_variables(file_path):
    """
    Extracts the game variables from file_path
    """
    game_variables = GameVariables()

    with open(file_path) as file:
        for line in file.readlines():
            if not line.rstrip() or line[0] == "#":
                continue
            var_type, variables = line.split(':')
            variables = list(map(str.rstrip, variables.split(',')))
            if var_type == "persons":
                game_variables.persons = variables
            elif var_type == "objects":
                game_variables.objects = variables
            elif var_type == "locations":
                game_variables.locations = variables
            elif var_type == "can_grab":
                variables = list(map(str.split, variables))
                game_variables.can_grab = dict()
                for variable in variables:
                    game_variables.can_grab[variable[0]] = variable[1:]
            elif var_type == "can_grab":
                variables = list(map(str.split, variables))
                game_variables.can_grab = dict()
                for variable in variables:
                    game_variables.can_grab[variable[0]] = variable[1:]

    return game_variables




def get_story_parameters(file_path):
    """
    Extracts the story parameters from file_path
    """
    story_parameters = StoryParameters()

    with open(file_path) as file:
        for line in file.readlines():
            if not line.rstrip() or line[0] == "#":
                continue
            parameter_type, parameters = line.split(':')
            parameters = list(map(str.rstrip, parameters.split(',')))
            if parameter_type == "samples":
                story_parameters.samples = int(parameters[0])
            elif parameter_type == "n_questions":
                story_parameters.n_questions = int(parameters[0])
            elif parameter_type == "seed":
                story_parameters.seed = int(parameters[0])
            elif parameter_type == "exhaustive":
                story_parameters.exhaustive = (parameters[0] == "True") # hack,, json form more clean
            elif parameter_type == "extra_exh_yes_no":
                story_parameters.extra_exh_yes_no = (parameters[0] == "True") # hack,, json form more clean
            elif parameter_type == "question_probability":
                story_parameters.question_probability = float(parameters[0])
            elif parameter_type == "actions_before_question":
                story_parameters.actions_before_question = int(parameters[0])
            elif parameter_type == "questions":
                story_parameters.questions = list(map(str.rstrip, parameters))
            elif parameter_type == "questions_distribution":
                story_parameters.questions_distribution = [float(str.rstrip(prob)) for prob in parameters]
            elif parameter_type == "max_actions":
                story_parameters.max_actions = int(parameters[0])
            elif parameter_type == "actions":
                story_parameters.actions = parameters
            elif parameter_type == "actions_distribution":
                story_parameters.actions_distribution = [float(str.rstrip(prob)) for prob in parameters]
            elif parameter_type == "coref":
                story_parameters.coref = parameters
            elif parameter_type == "coref_distribution":
                story_parameters.coref_distribution = [float(str.rstrip(prob)) for prob in parameters]
            elif parameter_type == "move":
                story_parameters.move = parameters
            elif parameter_type == "grab":
                story_parameters.grab = parameters
            elif parameter_type == "drop":
                story_parameters.drop = parameters
            elif parameter_type == "give":
                story_parameters.give = parameters
            elif parameter_type == "negate":
                story_parameters.negate = parameters
            elif parameter_type == "negate_distribution":
                story_parameters.negate_distribution = [float(str.rstrip(prob)) for prob in parameters]
            elif parameter_type == "coreference_prefixes":
                story_parameters.coreference_prefixes = parameters
            elif parameter_type == "entity_coreference_map":
                parameters = list(map(str.split, parameters))
                story_parameters.entity_coreference_map = dict()
                for parameter in parameters:
                    story_parameters.entity_coreference_map[parameter[0]] = parameter[1:]

        if story_parameters.questions_distribution == []:
            story_parameters.questions_distribution = [default_question_distribution(question) for question in story_parameters.questions]
        if story_parameters.actions_distribution == []:
            story_parameters.actions_distribution = [1 for action in story_parameters.actions]

    return story_parameters
