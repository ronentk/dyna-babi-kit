from typing import Tuple, List
from pathlib import Path
import tqdm
import pandas as pd
import logging
from collections import OrderedDict, defaultdict

from .helpers.event_calc import DECEvent, DECStory

logging.basicConfig(level = logging.INFO)

def extract_q_str(q_sent: str) -> str:
    return " ".join(q_sent.split("\t")[0].split(" ")[1:])

class StoryReader:
    def __init__(self, dec_file: Path, deep_load: bool = False):
        self.dec_file = dec_file
        self.stories_df = pd.read_json(str(dec_file), lines=True)
        self.dec_dict = OrderedDict()
        self.all_q_ids = []
        self.handle_dups = False
        if deep_load:
            self.deep_load()
            
        
    def deep_load(self):
        logging.info("Deep loading all stories in the split for quick future access. This may take a while...")
        for i in tqdm.tqdm(range(len(self.stories_df)), total=len(self.stories_df)):
            story_dict = self.stories_df.loc[i].to_dict()
            dec = DECStory.from_dict(story_dict)
            self.dec_dict[dec.uid] = dec
        self.all_q_ids = self._all_qids()
    
    def story_by_id(self, story_id: str) -> DECStory:
        story_df = self.stories_df[self.stories_df.uid == story_id]
        assert(len(story_df) == 1), f"Two stories with same id {story_id}"
        idx = story_df.index.item()
        story_dict = self.stories_df.loc[idx].to_dict()
        dec = DECStory.from_dict(story_dict)
        return dec
    
    def story_by_qid(self, qid: str) -> Tuple[DECStory, DECEvent]:
        story_id, q_id = qid.split("_")
        q_id = int(q_id)
        if story_id not in self.dec_dict:
            dec = self.story_by_id(story_id)
            self.dec_dict[story_id] = dec
        else:
            dec = self.dec_dict[story_id]
        q = dec.ev_by_timestep(q_id)
        assert(len(q) == 1), f"More than one event exists for timestep {q_id}: {q}!"
        q = q[0]
        return dec, q
    
    def _all_qids(self) -> List[str]:
        all_qids = []
        for uid, dec in self.dec_dict.items():
            qids = [f"{uid}_{t}" for t in dec.question_sent_idxs()]
            all_qids += qids
        return all_qids
    
    def all_qid_strs(self) -> List[Tuple[str, str]]:
        all_qids_q_strs = []
        for uid, dec in self.dec_dict.items():
            qids = [(f"{uid}_{t}", f"{dec.babi_story[t-1]}") for t in dec.question_sent_idxs()]
            all_qids_q_strs += qids
        return list(all_qids_q_strs)
            
