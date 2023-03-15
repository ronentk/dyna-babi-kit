
from typing import List, Dict
from pathlib import Path
import os
from shutil import copyfile
import argparse
import numpy.random as random
import numpy as np
import pickle
import json
import tqdm
import traceback
from collections import  defaultdict
from dataclasses import dataclass, field

from .game_variables_parser import get_story_parameters, get_game_variables
from .game_variables_parser import StoryParameters, GameVariables
from .world import World
from .Entities.Location import Location
from .Entities.Person import Person
from .Entities.Object import Object
from .Actions.ActionList import ActionList
from .Questions.QuestionList import QuestionList

from .story_filter import StoryFilter, FilterConfig, FilterBank
from .helpers.utils import RANDOM_SEED, seed, choice_np_rng
from .helpers.transformer_preproc import dec_story_to_transformer_inputs
from .helpers.sst.instance_sst import dec_to_sst_insts, SSTSampleOptions, InstanceSST, transformer_insts_from_sst, dec_to_sst_qa_insts
from .helpers.event_calc import DECStory, DECEvent, check_dec_answers_consistency
from .inference_engine import InferenceEngine



def match_sst_inst_to_filt_q(sst_inst: InstanceSST, dec: DECStory) -> DECEvent:
    """
    Due to filtering, an sst instnace question indices can be misaligned with 
    a respective filtered dec story question. This method attempts to find a 
    matching question and returns it if one exists.

    Parameters
    ----------
    sst_inst : InstanceSST
        DESCRIPTION.
    dec : DECStory
        DEC story with possibly filtered timesteps.

    Returns
    -------
    DECEvent
        The matching question event or None if no match found.

    """
    num_sents = len(sst_inst.texts)
    
    dec_story_sent_idxs = dec.story_sent_idxs()
    
    assert(len(dec_story_sent_idxs) >= num_sents)
    
    dec_q_idxs = dec.question_sent_idxs()
    corrected_q_idx = dec_story_sent_idxs[num_sents - 1] + 1
    
    if corrected_q_idx in dec_q_idxs:
        return dec.ev_by_timestep(corrected_q_idx)[0]
    else:
        return None
    

@dataclass
class StoryWriterConfig:
    out_dir: str
    var_files: List[str] = field(default_factory=lambda: [])
    param_file: str = None
    no_write: bool = False # for debugging - doesn't generate any output files
    write_dec: bool = False # write ouput in DEC format
    write_tt: bool = False # write output in TransformerInstance format
    use_new_engine: bool = False
    story_subsample_pct: float = 1
    seeds_file: str = None
    filter_config_file: str = None
    only_dev: bool = False
    # for generating data in breakpoint transformers format
    write_sst: bool = False 
    sst_to_vt: bool =  False
    sst_qa: bool = False # write questions in sst format (only supports WHERE_OBJ/WHERE_PERSON) currently
    sst_options: SSTSampleOptions = None
    
    

    def __post_init__(self):
        if self.var_files:
            self.var_files = [os.path.abspath(var_file) for var_file in self.var_files]
        if self.param_file:
            self.param_file = os.path.abspath(self.param_file)


    @property
    def out_path(self):
        return Path(self.out_dir)
    
    
    


def entities_from_vars(world, vars):
    """
    Creates Entities according to the attributes of a GameVariables object vars.
    :param world: A World object
    :param vars: A GameVariables object
    """
    locations = [Location(world, location) for location in vars.locations]
    persons = [Person(world, person) for person in vars.persons]
    objects = [Object(world, object) for object in vars.objects]
    return locations + persons + objects


def action_list_from_params(world, params):
    """
    Extracts actions from the StoryParameters object params.
    :param world: A World object
    :param params: A StoryParameters object
    """
    action_list = ActionList(world)
    action_list.init_from_params(params.actions, params.actions_distribution)
    return action_list


def question_list_from_params(world, params):
    """
    Extracts questions from the StoryParameters object params.
    :param world: A World object
    :param params: A StoryParameters object
    """
    question_list = QuestionList(world)
    question_list.init_from_params(params.questions, params.questions_distribution)
    return question_list


def add_sentences(data, sentences, sentence_idx):
    """
    Adds the list of sentences to data one by one, increasing sentence_idx appropriately
    """
    for sentence in sentences:
        text_part = sentence[0]
        sentence_str = str(sentence_idx) + " " + text_part + "\n"
        data.append(sentence_str)

        sentence_idx += 1
    return data, sentence_idx

def init_world(params, vars):
    """ 
    Initialize world with given parameters and variables.
    """
    world = World(params=params)
    entities = entities_from_vars(world, vars)
    world.populate(entities)
    action_list = action_list_from_params(world, params)
    question_list = question_list_from_params(world, params)
    world.rule(params, action_list, question_list)
    return world



class StoryWriter(object):
    """ 
    Manages generation and writing of bAbI stories to file.
    """
    def __init__(self, config: StoryWriterConfig,
                         story_parameters: StoryParameters = None,
                         game_variables: GameVariables = None,
                         filter_config: FilterConfig = None):
        self.config = config
        
        self.out_dir = Path(self.config.out_dir)
        
        # load game variables
        if game_variables:
            self.vars = game_variables
        else:
            self.vars = sum([get_game_variables(var_file) for var_file in self.config.var_files], GameVariables())
            
        # load story parameters
        if story_parameters:
            self.params = story_parameters
        else:
            self.params = get_story_parameters(self.config.param_file)
        
        # # set random seeds for reproduceability
        self.rseed = seed(self.params.seed)
        story_seeds_seed = np.random.randint(1, np.iinfo(np.int32).max)
        # seed for generating new story seeds
        self.story_seeds_rng = np.random.RandomState(story_seeds_seed)
        

        if self.config.only_dev:
            # generate only dev set
            self.n_samples = [self.params.samples]
            self.set_names = ["_valid"]
        else:
            # # number of samples for train, validation and test set
            self.n_samples = [self.params.samples, self.params.samples // 10, self.params.samples // 10]
            self.set_names = ["_train", "_valid", "_test"]
    
        
        # store stories in dec form
        self.dec_stories = {}
        self.dec_map_by_uid = defaultdict(lambda: defaultdict(list))
        
        # for storing instances in sst format
        self.sst_instances = defaultdict(list)
        self.sst_options = config.sst_options if config.sst_options else SSTSampleOptions()
        
        # storing vanilla transformers variants of sst instaces
        self.vt_sst_instances = defaultdict(list)
        
        # storing sst versions of questions
        self.sst_qa_instances = defaultdict(list)
        
        # vanilla transformers versions of prop based questions
        self.vt_prop_qa_instances = defaultdict(list) 
        
        if self.manual_seeding:
            self.seeds = json.load(Path(self.config.seeds_file).open())
            
        else:
            self.seeds = None
            
        self._sample_count = 0
            
        
        
        if filter_config:
            self.filtering = True
            self.story_filter = FilterBank(filter_config)
        else:
            if self.config.filter_config_file:
                self.filtering = True
                filter_config_path = Path(self.config.filter_config_file)
                self.story_filter = StoryFilter.from_filter_config_file(filter_config_path)
            else:
                self.filtering = False
                self.story_filter = StoryFilter()
        
        if self.config.story_subsample_pct < 1:
            self.filtering = True
        
        self._uids_to_write = defaultdict(lambda: defaultdict(list))

    
    @property
    def manual_seeding(self) -> bool:
        return self.config.seeds_file != None
    
    @property
    def no_write(self) -> bool:
        # return True if in no write mode
        return self.config.no_write
    
    def prepare_out_dir(self):
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # copy params file
        out_param_file = self.out_dir / "story_params.json"
        out_param_file.write_text(self.params.to_json())
        
        # copy variable files
        dest_path = self.out_dir / "game_vars.json"
        dest_path.write_text(self.vars.to_json())
            
        # copy filter config, if exists
        if self.filtering:
            dest_path = self.out_dir / "filter_config.json"
            dest_path.write_text(self.story_filter.config_json())
            
    def write_seeds(self):
        seeds = {}
        seeds["initial"] = self.rseed
        for split, stories in self.dec_stories.items():
            split = split.replace("_", "")
            seeds[split] = [story.seed for story in stories]
        
        seed_file = self.out_dir / "seeds.json"
        json.dump(seeds, seed_file.open(mode="w"))
    
    @property
    def uids_to_write(self):
        if not self._uids_to_write:
            return self.dec_map_by_uid
        else:
            return self._uids_to_write
        
    def subsample_uid(self, split: str, uid: str) -> bool:
        """ 
        For subsampling purposes. If subsampling, return True if seed included
        in list of seeds to be written, False o.w.
        """
        
        # remove suffixes if exist
        uid = uid.split("_")[0]
        
        return uid in self.uids_to_write[split]
        

    def subsample_stories(self, subsample_pct: float):
        """
        For each split, select `subsample_pct` uids to write (min=1).
        """ 
        for split, stories_map in sorted(self.dec_map_by_uid.items(), key=lambda x: x[0]):
            n_stories_to_take = int(np.ceil(len(stories_map) * subsample_pct))
            selected_uids = choice_np_rng(stories_map.keys(), self.story_seeds_rng, 
            n_stories_to_take, replace=False)
            self._uids_to_write[split] = {u: True for u in selected_uids}
    
    def build_world(self, params=None):
        params = self.params if not params else params
        world = init_world(params, self.vars)
        return world
    
    def write_dec_stories(self):
        for split, stories in self.dec_stories.items():
            wr_split = split
            if split == "valid":
                wr_split = "dev" # switch to conform to tt format
            dec_split_file = self.out_dir / (f"dec_{wr_split}.jsonl")
            json_stories = [dec_story.to_json() for dec_story in stories if self.subsample_uid(split, dec_story.uid)]
            dec_split_file.write_text("\n".join(json_stories))
            
    def write_tt_format(self):
        for split, stories in self.dec_stories.items():
            json_stories = []
            wr_split = split
            if split == "valid":
                wr_split = "dev" # switch to conform to tt format
            split_file = self.out_dir / (f"{wr_split}.jsonl")
            for dec_story in stories:
                if self.subsample_uid(split, dec_story.uid):
                    json_stories += [ti.to_json() for ti in 
                                         dec_story_to_transformer_inputs(dec_story)
                                         ]
            split_file.write_text("\n".join(json_stories))
            
    
    def write_babi_from_dec(self):
        for split, stories in self.dec_stories.items():
            babi_split_file = self.out_dir / (f"{split}.txt")
            babi_stories = ["".join(dec_story.babi_story) for dec_story in stories 
                            if self.subsample_uid(split, dec_story.uid)]
            babi_split_file.write_text("".join(babi_stories))
        
        
        
        
    
    def write_data(self):
        """
        Creates new data following the specifications in the config files given as arguments to the program
        :return:
        """
        
        print(f"Writing files to {str(self.out_dir)}")
        if not self.no_write:
            self.prepare_out_dir()

        for i in range(len(self.n_samples)):
            self.params.samples = self.n_samples[i]
            split = self.set_names[i].replace("_", "")
            print(f"Generating {split} split...")
            world = self.build_world(self.params)
    
            # reset filter in case counting types of stories generated        
            self.story_filter.reset()
            
    
            data = self.generate_data(world, self.params,
                                              self.params.exhaustive, split, use_new_engine=self.config.use_new_engine)
            
            
            samples = "".join(data)
            
            
            if not self.no_write:
                
                if not self.filtering:
                    out_fp = self.out_dir / (split + ".txt")
                    out_fp.write_text(samples)
        
                    

        if not self.no_write:

            # write seeds used for generation of each story, for reproduceability
            self.write_seeds()

            # if sub-sampling, select seeds to write out of generated stories
            if self.config.story_subsample_pct < 1:
                self.subsample_stories(self.config.story_subsample_pct)
            
            if self.filtering:
                print("Writing filtered stories in bAbI format...")
                self.write_babi_from_dec()
            
            if self.config.write_dec:
                print("Writing stories in DEC format...")
                self.write_dec_stories()
            
            if self.config.write_tt:
                print("Writing stories in TT format...")
                self.write_tt_format()
            
    
    
    
    @property
    def sample_count(self):
        if self.filtering:
            return self.story_filter.num_passed
        else:
            return self._sample_count
        
    def generate_data(self, world, params, exhaustive: bool = False,
                      split: str = None, use_new_engine: bool = False):
        """
        Generates a string of concatenated bAbI-style stories (data) according to the specifications in params and the
        members and attributes of world.
        :param world: A World object
        :param params: A StoryParameters object
        :param exhaustive: Whether to generate questions exhaustively, or not
        :param split: Split (train/test/valid) this story belongs to
        :return:
        """
        data = []
        used_seeds = set()
        seeds_counter = 0
        self.dec_stories[split] = []
        
        c = 0
        
        
        pbar = tqdm.tqdm(total=params.samples)
        self._sample_count = 0
        
        exhausted_search = False
        
        while self.sample_count < params.samples and not exhausted_search:
            current_count = self.sample_count
            if not self.manual_seeding:
                new_seed = self.story_seeds_rng.randint(1, np.iinfo(np.int32).max)
                # don't use same seed twice
                sample_seed = seed(new_seed, used_seeds)
                
            else:
                # use pre-loaded seed
                sample_seed = seed(self.seeds[split][len(used_seeds)])
            
            
            num_seeds = len(used_seeds)
            
            
    
            sentence_idx = 1
            story = []
            n_questions = 0
            question_gap = 0
    
            world.forget()
            world.allocate()
            world.current_seed = sample_seed
    
            while n_questions < params.n_questions:
                # may be exceeded for case exhaustive == True
                if question_gap >= params.actions_before_question:
                    q_p = np.random.uniform(0, 1)
                    if q_p < params.question_probability:
                        if world.can_ask():
                            questions, answers = world.ask(exhaustive=exhaustive)

                            
                            story, sentence_idx = add_sentences(story, questions, sentence_idx)
                            question_gap = 0
                            n_questions += len(questions)
                            continue
    
                sentences = None
                if world.can_act():
                    sentences = world.make_action()
                story, sentence_idx = add_sentences(story, sentences, sentence_idx)
                question_gap += len(sentences)
            
            dec_story = world.to_dec_story(story)

            # solve using InferenceEngine and update <dec_story>
            if use_new_engine:
                try:
                    engine = InferenceEngine()
                    last_q = list(world.q_history.keys())[-1]
                    for ind in range(1, last_q+1):
                        if ind in world.history:
                            engine.add_event(world.history[ind])
                        else:
                            ans, s_facts = engine.solve_question(world.q_history[ind])
                            dec_story.ie_answers[ind] = ans
                            dec_story.ie_s_facts[ind] = s_facts
                    
                    # if IE has different answer, go with it
                    check_dec_answers_consistency(dec_story)
                    
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print(f"Error with story: {dec_story.seed}: {str(dec_story)}")
            
            seeds_counter += 1
            
            
            # filter stories if configured
            if self.filtering:
                if self.story_filter.is_active:
                    passed_filter, filtered_dec_story = self.story_filter.filter_story(dec_story)
                    if passed_filter:
                        data += story
                        self.dec_stories[split].append(filtered_dec_story)
                        self.dec_map_by_uid[split][filtered_dec_story.uid] = filtered_dec_story

                                                
                        n_qs = self.sample_count - current_count # new qs
                        self._sample_count += n_qs
                        used_seeds.add(sample_seed)

                        pbar.update(n_qs)

                else:
                    # end search 
                    exhausted_search = True
                    
            else:
                data += story
                self.dec_stories[split].append(dec_story)
                self.dec_map_by_uid[split][dec_story.uid] = dec_story


                self._sample_count += 1
                used_seeds.add(sample_seed)
                pbar.update(1)
        
        if self.filtering:
            print(f"Number of unique q sigs: {self.story_filter.num_sigs}. Num unique seeds checked: {seeds_counter}. Exhausted search: {exhausted_search}")
            print(f"Filter stats: {self.story_filter.get_stats()}")
        return data

