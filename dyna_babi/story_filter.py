
from typing import Tuple, List, Set, Union, Dict
from pathlib import Path
import numpy as np
import tqdm
import json
import logging
from itertools import combinations
from collections import Counter
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field

from .helpers.event_calc import DECEvent, DECStory, unify_timestep_evs, filter_keep_timesteps

logging.basicConfig(level = logging.INFO)



ALL_Q_SET = set(["all"])
MAX_INT = int(np.iinfo(np.int32).max)

# if a filter doesn't pass a single valid story after MAX_FAIL attempts
# it will be de-activated
MAX_FAIL = 20000 

@dataclass_json
@dataclass
class FilterConfig:
    """
    Attributes
        ----------
        required_supp_fact_types : Set[str]
            Each generated question must include at least one supporting 
            fact of one of the types in `required_supp_fact_types`. 
            Types can be same types that appear in `story_parameters.actions`
        valid_q_types : Set[str]
            Each generated question must be of one of the specified types (default = all).
        num_supp_fact_range : Tuple[int,int]
            Tuple of [min,max], each generated question must have a number of supporting facts
            in [min,max]. 
        max_stories_per_sig : int
            Allow at most `max_stories_per_sig` of a given story signature (e.g., #MOVE(P_1,O_1)?where(P_1))
        always_pass_q_types : Set[str]
            Always pass questions of types included in `always_pass_q_types`, these don't count 
            towards `max_stories_per_sig`.
        remove_distractors : bool
            Remove all distractor sentences (not in supporting facts set). (default = False).
            
    """
    required_supp_fact_types: Set[str] = field(default_factory=set)
    valid_q_types: Set[str] = field(default_factory=lambda: ALL_Q_SET)
    num_supp_fact_range: Tuple[int,int] = (0,10000)
    max_stories_per_sig: int = MAX_INT
    always_pass_q_types: Set[str] = field(default_factory=set)
    max_len: int = MAX_INT
    max_pass: int = MAX_INT
    max_fail: int = MAX_FAIL
    min_paths: int = 0
    max_paths: int = MAX_INT
    filter_each_q: bool = True
    
    def __post_init__(self):
        self.valid_q_types = set(self.valid_q_types)
    
    def valid_supp_fact_num(self, x: int) -> bool:
        low, high = self.num_supp_fact_range
        return x >= low and x <= high
    
    def valid_supp_fact_types(self, supp_facts: List[str]) -> bool:
        """ 
        Return true if at least one input supp fact matches specifications
        """
        return len(set(supp_facts).intersection(self.required_supp_fact_types))
    
    def validate_q_type(self, q_type: str) -> bool:
        if self.valid_q_types == ALL_Q_SET:
            return True
        else:
            return q_type in self.valid_q_types
        
    def always_pass_q_type(self, q_type: str):
        return q_type in self.always_pass_q_types
        


class StoryFilter:
    """ 
    Handles filtering story-question pairs based on user-specified conditions.
    """
    def __init__(self, filter_config: FilterConfig = None):
        self.config = filter_config if filter_config else FilterConfig()
        self.q_sig_counter = Counter()   
        self.passed_count = 0
        self.failed_count = 0
        
    @classmethod
    def from_filter_config_file(cls, filter_config_file: str):
        config_path = Path(filter_config_file)
        config_json = config_path.read_text() if config_path else None
        filter_config = FilterConfig.from_json(config_json)
        return cls(filter_config)
        
    
    def reset(self):
        self.q_sig_counter = Counter() 
        self.passed_count = 0
        self.failed_count = 0
        

    def filter_pass_if_any(self, story: DECStory) -> bool:
        for q_idx in story.question_sent_idxs():
            if self.filter(story, q_idx):
                q_sig = story.q_sig_str(q_timestep=q_idx)
                self.q_sig_counter[q_sig] += 1
                q_ev = story.ev_by_timestep(q_idx)[0]
                q_ev.chosen_q = True
                return True, story
            
        return False, story

    def filter_each_q(self, story: DECStory) -> bool:
        keep_q_idxs = []
        conditional_keep_idxs = []
        story_idxs = story.story_sent_idxs(include_qs=False)
        for q_idx in story.question_sent_idxs():
            q_sig = story.q_sig_str(q_timestep=q_idx)
            if self.q_sig_counter[q_sig] < self.config.max_stories_per_sig:
                if self.filter(story, q_idx):
                    kind = story.ev_by_timestep(q_idx)[0].kind.value if not type(story.ev_by_timestep(q_idx)[0].kind) == str else story.ev_by_timestep(q_idx)[0].kind
                    # mark q as chosen by filter
                    q_ev = story.ev_by_timestep(q_idx)[0]
                    q_ev.chosen_q = True
                        
                    if not self.config.always_pass_q_type(kind):
                        self.q_sig_counter[q_sig] += 1
                        keep_q_idxs.append(q_idx)
                        
                    else:
                        # add the priviliged questions only if any non-privileged ones are found
                        conditional_keep_idxs.append(q_idx)
        
        if keep_q_idxs:
            all_keep_idxs = list(sorted(story_idxs + keep_q_idxs + conditional_keep_idxs))
            filtered_dec = filter_keep_timesteps(story, all_keep_idxs)   
            return True, filtered_dec
        else:
            return False, story
    
    def filter_story(self, story: DECStory) -> Tuple[bool, DECStory]:
        if self.config.filter_each_q:
            passed_filter, filtered_story = self.filter_each_q(story)
        else:
            passed_filter, filtered_story = self.filter_pass_if_any(story)
            
 
        
        if passed_filter:
            self.passed_count += 1
            self.failed_count = 0 # reset fail counter if we found a good story
        else:
            self.failed_count += 1
        
        return passed_filter, filtered_story
                
    @property
    def num_passed(self) -> int:
        return sum(self.q_sig_counter.values())
    
    @property
    def num_sigs(self) -> int:
        return len(self.q_sig_counter)
            
    
    @property
    def is_active(self) -> bool:
        return ((self.num_passed < self.config.max_pass) and 
                (self.failed_count < self.config.max_fail))
        
    
    def config_json(self) -> str:
        return self.config.to_json()
    
    def filter(self, story: DECStory, q_idx: int) -> bool:
        """
        Checks if a given story,q pair pass the defined filters.

        Parameters
        ----------
        story : DECStory
            input story.
        q_idx : int
            index of target q in story.

        Returns
        -------
        bool
            True if passed filter, False o.w.

        """
        q_ev = story.ev_by_timestep(q_idx)[0]
        unified_ev_list = [unify_timestep_evs(ts) for ts in story.ev_by_timesteps(q_ev.supporting_facts)]
        
        # bypass regular filtering for privileged q types
        if self.config.always_pass_q_type(q_ev.kind):
            return True
        
        # check if number of supporting facts in desired range
        ie_s_facts = story.ie_s_facts
        
        # use new engine for facts if exist
        if q_idx in ie_s_facts:
            min_s_facts = min([len(s) for s in ie_s_facts.get(q_idx)])
            if not self.config.valid_supp_fact_num(min_s_facts):
                return False
            
        elif not self.config.valid_supp_fact_num(len(unified_ev_list)):
            return False
        
        # check if question type is one of required types
        if not self.config.validate_q_type(q_ev.kind):
            return False
        
        # check story length (up until q and not not including qs) is leq `max_len`
        rel_story_sents = [idx for idx in story.story_sent_idxs() if idx < q_idx]
        if len(rel_story_sents) > self.config.max_len:
            return False
        
        # check that number of reasoning paths in required range
        ie_s_facts = story.ie_s_facts
        if q_idx in ie_s_facts:
            num_paths = len(ie_s_facts.get(q_idx))
            if num_paths > self.config.max_paths or num_paths < self.config.min_paths:
                return False

        
        # check if supporting fact types match on of required types
        if not self.config.required_supp_fact_types:
            # if none required
            return True
        else:
            # check that support composition of question contains required types
            support_comp = set()
            for supp_ev in unified_ev_list:
                support_comp.add(supp_ev.kind)
                if supp_ev.is_coref: 
                    support_comp.add("co_ref")
                if supp_ev.is_conj: 
                    support_comp.add("conj")
            req = set(self.config.required_supp_fact_types)
            
            # return True iff required types contained in support composition
            return req.intersection(support_comp) == req
            
    
        return False
    
    def get_stats(self) -> Dict:
        stats = {
            "num_passed": self.num_passed,
            "num_failed": self.failed_count,
            "is_active": self.is_active
            }
        return stats
                        
        
        

class FilterBank:       
    def __init__(self, filter_configs: Union[FilterConfig,List[FilterConfig]]):
        if not type(filter_configs) == list:
            self.configs = [filter_configs]
        else:
            self.configs = filter_configs
            
        self.filters = [StoryFilter(c) for c in self.configs]
        
    
    def config_json(self) -> str:
        data = {i: f.config.to_dict() for i, f in enumerate(self.filters)}
        return json.dumps(data)
    
    @property
    def num_passed(self) -> int:
        return sum([f.num_passed for f in self.filters])
    
    @property
    def num_sigs(self) -> int:
        return sum([f.num_sigs for f in self.filters])
    
        
    @property
    def is_active(self):
        return len(self.active_filters()) > 0
    
    
    def active_filters(self) -> List[StoryFilter]:
        """
        

        Returns
        -------
        List[StoryFilter]
            List of active StoryFilters.

        """
        active_filters = [f for f in self.filters if f.is_active]
        return active_filters

    def reset(self):
        for f in self.filters:
            f.reset()
            
    def get_stats(self) -> Dict:
        stats = {}
        for i, f in enumerate(self.filters):
            stats[i] = f.get_stats()
        return stats
            

    def filter_story(self, story: DECStory) -> Tuple[bool, DECStory]:
        for i, story_filter in enumerate(self.active_filters()):
            # pass story through each filter, break for first filter story passes
            # so we are only counting each story once
            passed_filter, filtered_story = story_filter.filter_story(story)
            if passed_filter:
                break
        
        return passed_filter, filtered_story
        
        
        
    