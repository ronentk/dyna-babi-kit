"""Script for parsing babi tasks and optionally modifying them by adding
questions.

Usage:
  solve_babi_tasks.py DATA_PATH [--task_configs=<task_configs> --odir=<odir>]
                                  [--no_decs --trim_over] [(-v | --verbose)]
  
  solve_babi_tasks.py (-h | --help)


Options:
  -h --help     Show this screen.
  -v --verbose  Verbose output.
  --task_configs=<task_configs>  Path to task specific config [default: none].
  --odir=<odir>  Output dir [default: none].
  --no_decs  Don't create DEC format files.
  --trim_over  Trim stories over max limit of 500 tokens.

"""
from typing import List, Dict, Any
import logging
import sys
import json
import os
from docopt import docopt
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dyna_babi.solver import solve_story, DumbSolver
from dyna_babi.game_variables_parser import StoryParameters, GameVariables
from dyna_babi.helpers.transformer_preproc import dec_story_to_transformer_inputs, TransformerInstance
from dyna_babi.helpers.event_calc import DECStory, DECEvent, filter_keep_timesteps

from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BABI_SU12_TASKS = ["qa1", "qa2", "qa3", "qa5", "qa6",
                   "qa7", "qa8", "qa9", "qa10", "qa11", "qa12", "qa13"]

# token length limit for babi instances for transformers
TOKEN_LENGTH_LIMIT = 500

def filter_long_dec(dec: DECStory, max_len: int = 500) -> DECStory:
    """ 
    Filter stories to remove questions over the maximum length handled by transformers.
    """
    
    # convert dec to list of transformer inputs and get their lengths
    tis = dec_story_to_transformer_inputs(dec)
    lengths = [len(tokenizer.tokenize(ti.input)) for ti in tis]
    
    assert(len(lengths) == len(dec.question_sent_idxs()))
    
    # for each instance pair length and idx of question
    len_pairs = list(zip(lengths, dec.question_sent_idxs()))
    
    
    keep = [t for l,t in len_pairs if l < 500]
    
    
    if keep:
        # max timestep to keep
        max_keep = max(keep)
        # create new filtered dec with only questions up to max_keep
        fdec = filter_keep_timesteps(dec, keep_idxs=range(1,(max_keep+1)))
    else:
        fdec = None
    
    filtered = len(lengths) - len(keep)
    return fdec, filtered
    
    

def check_over_length(tt_instance: TransformerInstance, len_limit: int = TOKEN_LENGTH_LIMIT) -> bool:
    """ 
    Return True if story above `len_limit`, False o.w.
    """
    

def write_tt_format(output_dir, dec_stories):
    for split, stories in dec_stories.items():
        json_stories = []
        wr_split = split
        if split == "valid":
            wr_split = "dev" # switch to conform to tt format
        split_file = output_dir / (f"{wr_split}.jsonl")
        for dec_story in stories:
            json_stories += [ti.to_json() for ti in \
                             dec_story_to_transformer_inputs(dec_story)
                                 ]
        split_file.write_text("\n".join(json_stories))
            
            
def write_dec_stories(output_dir, dec_stories):
    
    for split, stories in dec_stories.items():
        if split == "valid":
            split = "dev" # switch to conform to tt format
        dec_split_file = output_dir / (f"dec_{split}.jsonl")
        json_stories = [dec_story.to_json() for dec_story in stories]
        print(f"Writing {len(json_stories)} to {split}...")
        dec_split_file.write_text("\n".join(json_stories))

def write_babi_from_dec(output_dir, dec_stories):
    for split, stories in dec_stories.items():
        babi_split_file = output_dir / (f"{split}_filtered.txt")
        babi_stories = ["".join(dec_story.babi_story) for dec_story in stories]
        babi_split_file.write_text("".join(babi_stories))
        
@dataclass_json
@dataclass
class SolverConfig:
    tasks: List[str] = field(default_factory=lambda: BABI_SU12_TASKS)
    inject_config: Dict[str, Any] = field(default_factory=lambda: {})
    
    @classmethod
    def from_file(cls, fpath: str):
        fpath = Path(fpath)
        return SolverConfig.from_json(fpath.read_text())
    
    def check_inject(self, task: str) -> bool:
        return task in self.inject_config and "add_q_types" in self.inject_config[task]
    
    def inject_params(self, task: str):
        assert(self.check_inject(task))
        add_types = self.inject_config[task]["add_q_types"]
        orig_types = self.inject_config[task]["orig_q_types"]
        zipped = list(zip(orig_types, [0.0]*len(orig_types)))+ \
          list(zip(add_types, [1.0]*len(add_types)))
        questions, dist = zip(*zipped)
        return list(questions), list(dist)
    
    def to_pretty_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)
                     


if __name__ == '__main__':
    
    args = docopt(__doc__, version='0.0')
    
    babi_data_path = Path(args.get("DATA_PATH"))
    odir_path = args.get("--odir")
    task_configs = args.get("--task_configs")
    write_decs = not args.get("--no_decs")
    verbose = args.get("--verbose")
    trim_over_len = args.get("--trim_over")
    
    all_filtered = Counter()
    q_counts = Counter()
    
    if verbose:
        logger.setLevel(level=logging.DEBUG)
    
    if trim_over_len:
        logger.info("Loading tokenizer to check story length...")
        # any tokenizer will do, just for length checking
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = None
        
    tasks_config = SolverConfig() if task_configs == "none" else SolverConfig.from_file(task_configs)
    
    odir_path = Path(odir_path) if odir_path != "none" else None 
    
    if odir_path:
        logging.info(f"Creating output folder at {odir_path}...")
        odir_path.mkdir(parents=True, exist_ok=True)
        
    
    dec_stories = defaultdict(list)
    
    logger.info("Processing tasks...")
    
    for task in tqdm(tasks_config.tasks, total=len(tasks_config.tasks)):
        
        # check if we want to inject question types to task
        inject_qs = tasks_config.check_inject(task)
        if inject_qs:
            # get parameters for question injection
            qs, dist = tasks_config.inject_params(task)
            params = StoryParameters(questions=qs,
                                     questions_distribution=dist)
        else:
            params = StoryParameters()
        
        ds = DumbSolver(story_params=params, inject_qs=inject_qs)
        
        # load task data in babi format
        for data_file in babi_data_path.glob(f"{task}_*.txt"):
            logger.debug(f"Processing file {data_file}")
            with data_file as f:
                data_f = f.open()
                raw_data = data_f.readlines()
            
            split = data_file.stem.split("_")[-1]
            res = ds.solve_stories(raw_data, create_decs=True)
            
            # add task info
            for dec in res["decs"]:
                dec.task = task
                q_counts[split] += len(dec.question_sent_idxs())
                
            
            if trim_over_len:
                processed_decs, filtered_cnts = zip(*[filter_long_dec(dec, TOKEN_LENGTH_LIMIT) \
                                                      for dec in res["decs"]])
                all_filtered[split] += sum(filtered_cnts)
            else:
                processed_decs = res["decs"]
            
            dec_stories[split] += [p for p in processed_decs if p]
        
        
    if odir_path:
        
        config_path = odir_path / "solver_config.json"
        config_path.write_text(tasks_config.to_pretty_json())
        
        logger.debug(f"Writing stories in bAbI format to {str(odir_path)}...")
        write_babi_from_dec(odir_path, dec_stories)
        
        logger.debug(f"Writing stories in TT format to {str(odir_path)}...")
        write_tt_format(odir_path, dec_stories)
        
        if write_decs:
            logger.debug(f"Writing DEC stories to {str(odir_path)}...")
            write_dec_stories(odir_path, dec_stories)
        
    
    logger.info(f"Total question counts: {q_counts}")
    if trim_over_len:
        logger.info(f"Filtered stories over length limit of {TOKEN_LENGTH_LIMIT}: {all_filtered}")
    logger.info("Done!")
    
        
        
        
        
    
