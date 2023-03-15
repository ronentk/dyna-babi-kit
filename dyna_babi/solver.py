"""
Solves any bAbI task from 1-13 or any mix of them, using old-school procedural coding.
Note that the original bAbI task 5 contains some counter-intuitive answers to some questions.
The answers of this solver are the more coherent ones - we leave it to the reader to convince his or her self of that.
"""

import os
from typing import Optional, List, Tuple, Dict
import argparse
from pathlib import Path
import numpy.random as random
from tqdm import tqdm
import re
import logging

from dyna_babi.game_variables_parser import get_game_variables, get_story_parameters, GameVariables, StoryParameters
from dyna_babi.helpers.event_calc import DECStory
from dyna_babi.helpers.utils import renumber_story, reformat_lines
from dyna_babi.world import World
from dyna_babi.Entities.Location import Location
from dyna_babi.Entities.Person import Person
from dyna_babi.Entities.Object import Object
from dyna_babi.Actions.ActionList import ActionList
from dyna_babi.Questions.QuestionList import QuestionList
from .inference_engine import InferenceEngine

logging.basicConfig(level = logging.INFO)

ALL_STORY_TASKS = [1,2,3,5,6,7,8,9,10,11,12,13]
def entities_from_vars(world, vars):
    locations = [Location(world, location) for location in vars.locations]
    persons = [Person(world, person) for person in vars.persons]
    objects = [Object(world, object) for object in vars.objects]
    return locations + persons + objects


def action_list_from_params(world, params):
    action_list = ActionList(world)
    action_list.init_from_params(params.actions, params.actions_distribution)
    return action_list


def question_list_from_params(world, params):
    question_list = QuestionList(world)
    question_list.init_from_params(params.questions, params.questions_distribution)
    return question_list

def act_line(world, vars, last_persons, task, idx, line):
    """
    if-elif-else to determine what the sentence is.
    split the sentence. extract entities names, then actual entities.
    do_specific action or question based on extracted data.
    override = True will allow do_specific to ignore the current state of
    entities in the game world, and instead treat them as if they were in the state implied by the story
    """
    params = world.params
    coreferences = [" " + values[0] + " " for values in params.entity_coreference_map.values()]

    answer = None
    guess = None
    is_q = False
    is_correct = False

    words = re.findall(r"[\w']+", line)

    if '\t' in line:
        # question
        is_q = True
        if "was" in line:
            # where was
            object = world.get_entity_by_name(words[4])
            location = world.get_entity_by_name(words[7])
            answer = words[8]
            _, guess = world.get_question_by_kind("where_was_object").ask_specific(object, location)
        elif "Where" in line and "is the" in line:
            # where object
            object = world.get_entity_by_name(words[4])
            answer = words[5]
            _, guess = world.get_question_by_kind("where_object").ask_specific(object)
        elif "Where" in line:
            # where person
            person = world.get_entity_by_name(words[3])
            answer = words[4]
            _, guess = world.get_question_by_kind("where_person").ask_specific(person)
        elif "How many" in line:
            # counting
            person = world.get_entity_by_name(words[5])
            answer = words[7]
            _, guess = world.get_question_by_kind("counting").ask_specific(person)
        elif "carrying" in line:
            # listing
            person = world.get_entity_by_name(words[3])
            answers = [word for i, word in enumerate(words) if (i > 4) and word.isalpha()]
            answers.sort()
            answer_str = "".join([answer + "," for answer in answers])
            answer = answer_str[:-1]
            _, guess = world.get_question_by_kind("list").ask_specific(person)
        elif "Is" in line:
            # YesNo
            person = world.get_entity_by_name(words[2])
            location = world.get_entity_by_name(words[5])
            answer = words[6]
            _, guess = world.get_question_by_kind("yes_no").ask_specific(person, location)
            a = 1
        else:
            # giving
            # original giving task seems to be buggy - the answer to a "what_give" questions is the first object
            # that was passed between the pair, instead of the last. my version is the correct one.
            if "gave" in line and " to " in line:
                object = world.get_entity_by_name(words[4])
                person2 = world.get_entity_by_name(words[6])
                answer = words[7]
                _, guess = world.get_question_by_kind("giving").ask_specific(None, person2, object, "gave_to")
            elif "gave" in line:
                object = world.get_entity_by_name(words[4])
                answer = words[5]
                _, guess = world.get_question_by_kind("giving").ask_specific(None, None, object, "gave")
            elif "received" in line:
                object = world.get_entity_by_name(words[4])
                answer = words[5]
                _, guess = world.get_question_by_kind("giving").ask_specific(None, None, object, "received")
            elif "Who" in line and "give" in line:
                person1 = world.get_entity_by_name(words[3])
                object = world.get_entity_by_name(words[6])
                answer = words[8]
                _, guess = world.get_question_by_kind("giving").ask_specific(person1, None, object, "who_give")
            elif "What" in line and "give" in line:
                person1 = world.get_entity_by_name(words[3])
                person2 = world.get_entity_by_name(words[6])
                answer = words[7]
                _, guess = world.get_question_by_kind("giving").ask_specific(person1, person2, None, "what_give")
    else:
        coref = False
        # action
        if [move for move in params.move if move in line]:
            # move, coref, conj, compound,
            if [coref for coref in coreferences if coref in line]:
                # move coref
                coref = True
                person = last_persons[0]
                location = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("move").act_specific(person, location, override=True, coref=coref)
            elif "they" in line:
                # compound
                coref = True
                location = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("conj").act_specific(last_persons[0], last_persons[1], location, override=True, coref=coref)
            elif len([person for person in vars.persons if person in line]) > 1:
                # conj
                person1 = world.get_entity_by_name(words[1])
                person2 = world.get_entity_by_name(words[3])
                location = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("conj").act_specific(person1, person2, location, override=True, coref=coref)
                last_persons.clear()
                last_persons.extend([person1, person2])
            else:
                # move
                person = world.get_entity_by_name(words[1])
                location = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("move").act_specific(person, location, override=True)
                last_persons.clear()
                last_persons.extend([person])
        elif [grab for grab in params.grab if grab in line]:
            # grab, coref
            if [coref for coref in coreferences if coref in line]:
                # coref
                coref = True
                try:
                    words.remove("there")
                    words.remove("up")
                    
                except ValueError:
                    pass
                person = last_persons[0]
                object = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("grab").act_specific(person, object, override=True, coref=coref)
            else:
                #grab
                try:
                    words.remove("there")
                    words.remove("up")
                except ValueError:
                    pass
                person = world.get_entity_by_name(words[1])
                object = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("grab").act_specific(person, object, override=True)
                last_persons.clear()
                last_persons.extend([person])
        elif [drop for drop in params.drop if drop in line]:
            # grab, coref
            if [coref for coref in coreferences if coref in line]:
                # coref
                coref = True
                try:
                    words.remove("there")
                    words.remove("down")
                except ValueError:
                    pass
                person = last_persons[0]
                object = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("drop").act_specific(person, object, override=True, coref=coref)
            else:
                # drop
                try:
                    words.remove("there")
                    words.remove("down")
                except ValueError:
                    pass
                person = world.get_entity_by_name(words[1])
                object = world.get_entity_by_name(words[-1])
                world.get_action_by_kind("drop").act_specific(person, object, override=True)
                last_persons.clear()
                last_persons.extend([person])
        elif [give for give in params.give if give in line]:
            # give
            person1 = world.get_entity_by_name(words[1])
            person2 = world.get_entity_by_name(words[6])
            object = world.get_entity_by_name(words[4])
            world.get_action_by_kind("give").act_specific(person1, person2, object, override=True)
            last_persons.clear()
            last_persons.extend([person1])
        elif "either" in line:
            person = world.get_entity_by_name(words[1])
            location1 = world.get_entity_by_name(words[6])
            location2 = world.get_entity_by_name(words[9])
            world.get_action_by_kind("indef").act_specific(person, location1, location2, override=True)
            last_persons.clear()
            last_persons.extend([person])
        elif "is" in line:
            # negate
            person = world.get_entity_by_name(words[1])
            location = world.get_entity_by_name(words[-1])
            negate = ""
            for neg in params.negate:
                if neg in line:
                    negate = neg
                    break
            world.get_action_by_kind("negate").act_specific(person, negate, location, override=True)
            last_persons.clear()
            last_persons.extend([person])

    if is_q:
        if answer is not None and guess is not None and guess == answer:
            is_correct = True
        else:
            print("Mismatch in task {} in line {}: answer is {}, but guess was {}.".format(task, idx, answer, guess))
    return is_q, is_correct


def get_lines(task, tasks_dir: str = None):
    if not tasks_dir:
        tasks_dir = "../babi_data/tasks_1-20_v1-2/en-valid-10k"
    files = os.listdir(tasks_dir)
    files = [os.path.join(tasks_dir, f) for f in files]
    s = "qa{}_".format(task)
    train_file = [f for f in files if s in f and 'train' in f][0]
    valid_file = [f for f in files if s in f and "_valid" in f][0]
    test_file = [f for f in files if s in f and "_test" in f][0]

    train, valid, test = [], [], []
    with open(train_file, "r") as train_file:
        train = train_file.readlines()
    with open(valid_file, "r") as valid_file:
        valid = valid_file.readlines()
    with open(test_file, "r") as test_file:
        test = test_file.readlines()

    return train, test, valid

def write_dec_stories(output_dir, dec_stories):
    
    for split, stories in dec_stories.items():
        if split == "valid":
            split = "dev" # switch to conform to tt format
        dec_split_file = output_dir / (f"dec_{split}.jsonl")
        json_stories = [dec_story.to_json() for dec_story in stories]
        print(f"Writing {len(json_stories)} to {split}...")
        dec_split_file.write_text("\n".join(json_stories))
            
def solve_babi(vars, params, tasks, tasks_dir, dec_output_path):
    write_dec = dec_output_path != ""
    if write_dec:
        output_dir = Path(dec_output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
    dec_stories = {"train": [],
                   "valid": [],
                   "test": []}
    
    world = World(params=params)

    entities = entities_from_vars(world, vars)
    world.populate(entities)

    action_list = action_list_from_params(world, params)
    question_list = question_list_from_params(world, params)
    world.rule(params, action_list, question_list)

    for task in tasks:
        print(f"Processing task {task}...")
        train, valid, test = get_lines(task, tasks_dir)
        splits = [("train", train), ("valid", valid), ("test", test)]

        questions = 0
        correct = 0

        
        
        for split_name, split in splits:
            last_persons = []
            story_lines = []
            
            for i, line in tqdm(enumerate(split), total=len(split)):
                world.timestep = int(line.split()[0])
                if line.split()[0] == "1":
                    if story_lines and write_dec:
                        # if not first line of file
                        dec = world.to_dec_story(story_lines)
                        dec_stories[split_name].append(dec)
                        story_lines = []
                        last_persons = []
                    else:
                        story_lines = []
                        last_persons = []
                    world.forget()
                    world.allocate()
                    world.timestep = 1
                    
                is_q, is_correct = act_line(world, vars, last_persons, task, i, line)
                story_lines.append(line)
                if is_q:
                    questions += 1
                    if is_correct:
                        correct += 1
            
            # add last story
            dec = world.to_dec_story(story_lines)
            dec_stories[split_name].append(dec)
            
        print("correct: {} out of {}".format(correct, questions))
        
    if write_dec:
        print(f"Writing DEC stories to {str(output_dir)}...")
        write_dec_stories(output_dir, dec_stories)
        print("Done!")
        return dec_stories
    else:
        return []


def solve_story(vars, params, lines, write_dec: bool = False):
    world = World(params=params)

    entities = entities_from_vars(world, vars)
    world.populate(entities)

    action_list = action_list_from_params(world, params)
    question_list = question_list_from_params(world, params)
    world.rule(params, action_list, question_list)
    
    task = 0
    questions = 0
    correct = 0
    
    last_persons = []
    story_lines = []
    dec_stories = []
    
    for i, line in tqdm(enumerate(lines), total=len(lines)):
        world.timestep = int(line.split()[0])
        if line.split()[0] == "1":
            if story_lines and write_dec:
                # if not first line of file
                dec = world.to_dec_story(story_lines)
                dec_stories.append(dec)
                story_lines = []
                last_persons = []
            else:
                story_lines = []
                last_persons = []
            world.forget()
            world.allocate()
            world.timestep = 1
            
        is_q, is_correct = act_line(world, vars, last_persons, task, i, line)
        story_lines.append(line)
        if is_q:
            questions += 1
            if is_correct:
                correct += 1
    
    # add last story
    dec = world.to_dec_story(story_lines)
    dec_stories.append(dec)
            
    if write_dec:
        return questions, correct, dec_stories
    else:
        return questions, correct, []

def reset_world(world):
    world.forget()
    world.allocate()
    world.timestep = 1
    return world


def init_world(params: StoryParameters, vars: GameVariables):
    world = World(params=params)

    entities = entities_from_vars(world, vars)
    world.populate(entities)

    action_list = action_list_from_params(world, params)
    question_list = question_list_from_params(world, params)
    world.rule(params, action_list, question_list)
    return world


class DumbSolver:
    """
    Rule-based solver for bAbI stories. Solves any bAbI task from 1-13 or any mix of them, using old-school procedural coding.
    """
    def __init__(self, story_params: Optional[StoryParameters] = None, 
                 game_vars: Optional[GameVariables] = None,
                 inject_qs: bool = False):
        self.params = story_params if story_params else StoryParameters()
        self.vars = game_vars if game_vars else GameVariables()
        self.world = init_world(self.params, self.vars)
        self.dec_stories = []
        
        self.inject_qs = inject_qs
    
    def solve_story(self, lines: List[str], create_dec: bool = False) -> Dict:
        """
        Solves story provided as input in bAbI format.
        

        Parameters
        ----------
        lines : List[str]
            Story lines in bAbI format.
        create_dec : bool, optional
            Create and return the story in structured format. The default is False.

        Returns
        -------
        Dict
            Contains results of running solver on story, on following keys:
                - num_q: number of total questions.
                - num_correct: num. correct answers out of num_q.
                - dec: story in structured format, if specified.

        """
        last_persons = []
        story_lines = []
        questions = 0
        task = 0
        correct = 0
        res = {}
        
        world = self.world
        self.world = reset_world(self.world)
        
        for i, line in enumerate(lines):
            world.timestep = int(line.split()[0])
            is_q, is_correct = act_line(world, self.vars, last_persons, task, i, line)
            story_lines.append(line)
            if is_q:
                questions += 1
                if self.inject_qs:
                    all_qs = list(world.ask(exhaustive=True))
                    if all_qs:
                        q_lines = [q[0]+"\n" for q in all_qs[0]]
                        questions += len(all_qs)
                        story_lines += q_lines
                if is_correct:
                    correct += 1
        
        # go over all lines again after questions added
        if self.inject_qs:
            questions = 0
            task = 0
            correct = 0
            
            # renumber lines
            renumbered_lines = renumber_story(story_lines)
            
            
            
            self.world = reset_world(self.world)
            new_story_lines = []
            for i, line in enumerate(renumbered_lines):
                world.timestep = int(line.split()[0])
                is_q, is_correct = act_line(world, self.vars, last_persons,
                                            task, i, line)
                new_story_lines.append(line)
                if is_q:
                    questions += 1
                    if is_correct:
                        correct += 1
            story_lines = new_story_lines
            
            # reformat lines
            story_lines = reformat_lines(story_lines)
    
        # add last story
        if create_dec:
            dec = world.to_dec_story(story_lines)

            # try solving with new inference engine
            try:
                engine = InferenceEngine()
                last_q = list(world.q_history.keys())[-1]
                for ind in range(1, last_q+1):
                    if ind in world.history:
                        engine.add_event(world.history[ind])
                    else:
                        ans, s_facts = engine.solve_question(world.q_history[ind])
                        dec.ie_answers[ind] = ans
                        dec.ie_s_facts[ind] = s_facts
            except Exception as e:
                print(e)
                print(f"Error with story: {dec.seed}: {str(dec)}")



            res["dec"] = dec
        
        res["num_q"] = questions
        res["num_correct"] = correct
        
        return res
        
        
        
        
        
    
    def solve_stories(self, lines: List[str], create_decs: bool = False) -> Dict:
        """
        Solve sequence of stories provided in bAbI format.
        

        Parameters
        ----------
        lines : List[str]
            Stories lines in bAbI format.
        create_dec : bool, optional
            Create and return the story in structured format. The default is False.

        Returns
        -------
        Dict
            Contains results of running solver on stories, on following keys:
                - num_q: number of total questions.
                - num_correct: num. correct answers out of num_q.
                - decs: List of stories in structured format, if specified.

        """
        all_res = {"num_q": 0,
               "num_correct": 0,
               "decs": []}
        starts = [i for i,l in enumerate(lines[:-1]) if l.split()[0] == "1"] + [len(lines)]
        
        # break into list of n sublists, each containing one story's lines
        all_lines = [lines[starts[i]:starts[i+1]] for i in
                     range(len(starts)-1)]
        
        # solve stories
        for story_lines in tqdm(all_lines, total=len(all_lines)):
            res = self.solve_story(story_lines, create_decs)
            if create_decs:
                all_res["decs"].append(res["dec"])
            all_res["num_q"] += res["num_q"]
            all_res["num_correct"] += res["num_correct"]
        
        
        
        
        return all_res
    
    def solve_file(self, file_path: Path, out_dir: Path = None) -> Dict:
        """
        Load and solve file of stories in bAbI format.

        Parameters
        ----------
        file_path : Path
            Input file.
        out_dir : Path, optional
            Directory to save output file under, using same name as original. The new file will be of format <old_file_name>_dec.jsonl.

        Returns
        -------
        res : Dict
            Same as `solve_stories`.

        """
        out_dir = Path(out_dir)
        
        logging.info(f"Starting processing: {file_path}...")
        with file_path.open() as f:
            all_lines = f.readlines()
        
        write_decs = out_dir != None
        
        res = self.solve_stories(all_lines, create_decs=write_decs)
        logging.info(f"Processed {res.get('num_q')} questions , {res.get('num_correct')} correct.")
        
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            dec_split_file = out_dir / (f"{file_path.stem}_dec.jsonl")
            json_stories = [dec_story.to_json() for dec_story in res["decs"]]
            logging.info(f"Writing {len(json_stories)} stories to {dec_split_file}...")
            dec_split_file.write_text("\n".join(json_stories))
            
        
        return res
            
    def solve_dir(self, in_dir: Path, out_dir: Path = None):
        """
        Solve all files in the input directory.

        Parameters
        ----------
        in_dir : Path
            Input directory containing txt files in bAbI format.
        out_dir : Path, optional
            Dir in which outputs will be created. If non specified, no files will be created.

        Returns
        -------
        all_res : Dict
             Dict of result dicts, keyed by file name.

        """
        in_dir = Path(in_dir)
        logging.info(f"Starting processing dir: {in_dir}...")
        all_res = {}
        for f in in_dir.glob("*.txt"):
            res = self.solve_file(f, out_dir)
            all_res[f.stem] = res
        return all_res
        
            
        
    
if __name__ == "__main__":
    pass