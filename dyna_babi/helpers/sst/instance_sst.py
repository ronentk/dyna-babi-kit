#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:53:35 2021

@author: ronent
"""
from typing import List, Tuple, Dict, Optional
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field

from ...Questions.Question import QuestionType
from ...helpers.transformer_preproc import dec_story_to_transformer_inputs, TransformerInstance
from ...world import World
from ..event_calc import DECStory, DECEvent, from_world_event
from ..transformer_preproc import strip_sents, process_question_sent
from .proposition import ProbProposition
from .belief_base import BeliefBase, get_obj_belief_timeline

import shortuuid

label_map = {
    1.0: "yes",
    0.5: "maybe",
    0: "no"
    }

@dataclass_json
@dataclass
class InstanceSST:
    """
    Used for generating data in situation modeling format (SST)
    """
    texts: List[List[str]] # List of sentences or arbitrary pieces of text to be separated by [SIT] token
    outputs: List[List[float]] # list of list of confidences pairs. Length should equal number of steps in story
    # list of list of tokenized props. Outer list length should equal number of steps in story,
    # inner list should match outputs s.t. zip(prop_lists, outputs) yields each prop and
    # associated belief score
    prop_lists: List[List[int]] 
    question: str
    answer: str
    supporting_facts: List[int]
    prop_times: List[List[int]] # same dimensions as `prop_lists`; for each prop, its associated time
    uid: str
    seed: int
    q_sub_id: int
    guid: str = None
    task: str = ""
    
    
    def __post_init__(self):
        self.check_valid()
        self.guid = self.q_uid
    
    def check_valid(self):
        assert(len(self.texts) == len(self.prop_lists)), f"len(texts) = {len(self.texts)} != len(prop_lists) = {len(self.prop_lists)}"
        assert(len(self.outputs) == len(self.prop_lists))
        assert([len(l) for l in self.outputs] == [len(l) for l in self.prop_lists])
        
    @property
    def q_uid(self):
        return self.uid + f"_{self.q_sub_id}"
    
    @property
    def target_props(self):
        return [list(zip(self.prop_lists[i], self.outputs[i])) for i in  range(len(self.outputs))]

@dataclass_json
@dataclass
class SSTSampleOptions:
    diff_props: bool = True # include props whose belief changed
    n_sample_pos: int = 2 # number of positive propositions to sample at each timestep 
    n_sample_neg: int = 2 # number of negative propositions to sample at each timestep
    max_per_t: int = 5 #  # max. number of propositions to sample at each timestep
    verbose_props: bool = False # if True, represent props in string form, use prop idx o.w.
    
        

def create_target_prop_list(texts: List[str], until_t: int, seed: int,  world: World, diff_props: bool = True,
                            n_sample_pos: int = 2, n_sample_neg: int = 2, max_per_t: int = 5,
                            verbose_props: bool = False) -> List[List[Tuple[int, float]]]:
    """
    

    Parameters
    ----------
    texts : List[str]
        DESCRIPTION.
    prop_lists : Dict
        DESCRIPTION.

    Returns
    -------
    List[List[Tuple[int, float]]]
        DESCRIPTION.

    """
    beliefs = BeliefBase(world.diff_props, seed=seed)
    target_prop_dict = beliefs.sample_props(until_t-1, diff_props=diff_props,
                                            n_sample_pos=n_sample_pos, n_sample_neg=n_sample_neg,
                                            max_per_t=max_per_t)
    target_prop_list = []
    
    for ts, pprops in target_prop_dict.items():
        if verbose_props:
            target_prop_list.append([(prop, belief, t) for prop, belief, t in pprops])
        else:
            target_prop_list.append([(world.prop2idx[prop], belief, t) for prop, belief, t in pprops])
    
    assert(len(target_prop_list) == len(texts)), f"Error in seed {seed}: target_prop_list length= {len(target_prop_list)}, len(texts)={len(texts)}, tpl={target_prop_list}, target_prop_dict={target_prop_dict}, texts = {texts}"
    
    return target_prop_list

def dec_to_sst_insts(dec_story: DECStory, world: World, sst_opts: SSTSampleOptions = None) -> List[InstanceSST]:
    """
    

    Parameters
    ----------
    world : World
        DESCRIPTION.

    Returns
    -------
    InstanceSST
        DESCRIPTION.

    """
    sst_opts = sst_opts if sst_opts else SSTSampleOptions()
    
    sst_insts = []
    stripped_lines = strip_sents(dec_story)
    
    # to account for question idxs getting dropped
    correction_map = {t: i for i,t in enumerate(dec_story.story_sent_idxs())}

    for q_timestep in dec_story.question_sent_idxs():
        q_idx = q_timestep - 1
        uid = dec_story.uid
        seed = dec_story.seed
        q_ev = dec_story.ev_by_timestep(q_timestep)[0]
        
        # get supporting facts
        supp_facts = q_ev.supporting_facts
        
        story_lines = [line for line in stripped_lines[:q_idx] if not "?" in line]
        
        question, answer = process_question_sent(stripped_lines[q_idx])
        
        target_prop_list = create_target_prop_list(story_lines, q_timestep, seed=seed, world=world, **sst_opts.to_dict())
        
        # unpack to list of props and list of outputs
        unpacked = [list(zip(*pairs)) for pairs in target_prop_list]
        prop_lists = [list(p[0]) for p in unpacked]
        outputs = [list(p[1]) for p in unpacked]
        times = [list(p[2]) for p in unpacked]
        
        # correct times due to dropping idxs of questions
        corrected_times = [[correction_map[p] for p in l] for l in times]
        
        
        inst = InstanceSST(texts=story_lines, outputs=outputs, prop_lists=prop_lists,
                           prop_times=corrected_times,
                    question=question, answer=answer, supporting_facts=supp_facts,
                    uid=uid, seed=seed, q_sub_id=q_timestep, task=dec_story.task)
        sst_insts.append(inst)
    
    return sst_insts


def get_where_was_obj_prop_lists(world, q_idx) -> List[List[Tuple[str, float, int]]]:
    """
    

    Parameters
    ----------
    world : TYPE
        DESCRIPTION.
    q_idx : TYPE
        DESCRIPTION.

    Returns
    -------
    List[List[Tuple[str, float, int]]]
        DESCRIPTION.

    """
    q_ev = world.q_history[q_idx]
    # assuming all events in story (that q is at end of story)
    assert(q_idx > max(world.history.keys()))
    assert(q_ev.kind == QuestionType.WHERE_WAS_OBJ)
    
    obj_name = q_ev.source
    loc_a = q_ev.target[0]
    loc_b = q_ev.ternary
    
    obj_timeline = get_obj_belief_timeline(obj_name, world.diff_props)
    
    locs = {i: p.args[1] for i, (t,p) in enumerate(obj_timeline)}
    times = {i: t for i, (t,_) in enumerate(obj_timeline)}
    
    # find occurences of location b
    occurences = [i for i,l in locs.items() if l == loc_b]
    
    # find all occurences s.t. loc a is followed by loc b
    matches = [(times.get(idx-1), times.get(idx)) for idx in occurences if locs.get(idx-1,"") == loc_a]
    
    # there should only be one transition from loc a to loc b
    if len(matches) > 1: 
        print(f"More than one possible answer to question: {matches}, taking last...")
    
    loc_a_start, loc_b_start = matches[-1]
    
    
    prop_lists = []
    for i in range(len(world.history)):
        t = i+1
        if t < loc_a_start:
            prop_lists.append([])
        elif t >= loc_a_start and t < loc_b_start:
            # make positive proposition for answer
            pos_prop = f"{obj_name} at {loc_a}"
            
            # make negative propositions for untrue facts
            other_locs = [l.name for l in world.locations if not l.name==loc_a]
            neg_props = [(f"{obj_name} at {l}", 0, 0) for l in other_locs]
            
            props_at_t = [(pos_prop, 1, 0)] + neg_props # 0 = dummy timestep
            
            prop_lists.append(props_at_t)
        elif t == loc_b_start:
            # make positive proposition for answer
            pos_prop = f"{obj_name} at {loc_b}"
            
            # make negative propositions for untrue facts
            other_locs = [l.name for l in world.locations if not l.name==loc_b]
            neg_props = [(f"{obj_name} at {l}", 0, 0) for l in other_locs]
            
            props_at_t = [(pos_prop, 1, 0)] + neg_props # 0 = dummy timestep
            
            prop_lists.append(props_at_t)
        else:
            assert(t > loc_b_start)
            prop_lists.append([])
        
    assert(len(prop_lists) == len(world.history))
    return prop_lists
    
    
    
    
    
    

def dec_to_sst_qa_insts(dec_story: DECStory, world: World, sst_opts: SSTSampleOptions = None) -> List[InstanceSST]:
    """
    For each question in the story, generate an SST instance.
    This instance will query the last sentence based on the type of question.
    Currently supported question types: {`where_object`, `where_person`}

    Parameters
    ----------
    dec_story : DECStory
        DESCRIPTION.
    world : World
        DESCRIPTION.

    Returns
    -------
    List[InstanceSST]
        DESCRIPTION.

    """
    sst_opts = sst_opts if sst_opts else SSTSampleOptions()
    sst_insts = []
    stripped_lines = strip_sents(dec_story)
    

    for q_timestep in dec_story.question_sent_idxs():
        q_idx = q_timestep - 1
        uid = dec_story.uid
        seed = dec_story.seed
        q_ev = dec_story.ev_by_timestep(q_timestep)[0]
        
        if not q_ev.kind in [QuestionType.WHERE_PERSON, QuestionType.WHERE_OBJ,
                             QuestionType.WHERE_WAS_OBJ]:
            continue
        
        # get supporting facts
        supp_facts = q_ev.supporting_facts
        
        story_lines = [line for line in stripped_lines[:q_idx] if not "?" in line]
        
        question, answer = process_question_sent(stripped_lines[q_idx])
        
        source = q_ev.source[0] # person or object name
        target = q_ev.target[0] # location name
        
        # make positive proposition for answer
        pos_prop = f"{source} at {target}"
        
        # make negative propositions for untrue facts
        other_locs = [l.name for l in world.locations if not l.name==target]
        neg_props = [(f"{source} at {l}", 0, 0) for l in other_locs]
            
        props_at_t = [(pos_prop, 1, 0)] + neg_props # 0 = dummy timestep
        
        if q_ev.kind in [QuestionType.WHERE_PERSON, QuestionType.WHERE_OBJ]:
            
            
            prop_lists = [[] for i in range(len(story_lines))]
            prop_lists[-1] += props_at_t
        
        elif q_ev.kind == QuestionType.WHERE_WAS_OBJ:
            prop_lists = get_where_was_obj_prop_lists(world,q_timestep)
        
        
        # unpack to list of props and list of outputs
        unpacked = [list(zip(*p)) if p else [] for p in prop_lists]
        prop_lists = [list(p[0]) if p else [] for p in unpacked]
        outputs = [list(p[1]) if p else [] for p in unpacked]
        times = [list(p[2]) if p else [] for p in unpacked]
        
        inst = InstanceSST(texts=story_lines, outputs=outputs, prop_lists=prop_lists,
                           prop_times=times,
                    question=question, answer=answer, supporting_facts=supp_facts,
                    uid=uid, seed=seed, q_sub_id=q_timestep, task=dec_story.task)
        sst_insts.append(inst)
        
    return sst_insts
    

def number_sents(sents: List[str], start: int = 1):
    return [f"{start+i} {sent}" for i,sent in enumerate(sents)]

def number_events(events: List[List[DECEvent]], start: int = 1):
    for i, t_events in enumerate(events):
        curr_t = i + start
        for ev in t_events:
            ev.timestep = curr_t
    return events
        
def prop_to_q(world, prop: str, out: float, t: int) -> DECEvent:
    """ 
    Convert proposition string to QuestionEvent format
    """
    answer = label_map.get(out)
    arg1, rel, arg2 = prop.split(" ")
    if rel == "holds":
        q_ev = DECEvent(kind=QuestionType.YES_NO_PERS_OBJ,
                             timestep=t, source=[arg1], target=[answer],
                             ternary=[arg2], is_q=True)
        q_str = f"Is {arg1} holding the {arg2}?\t{answer}"
    elif rel == "at":
        if world.ent_map.get(arg1) == "person" and world.ent_map.get(arg2) == "location":
            q_ev = DECEvent(kind=QuestionType.YES_NO,
                             timestep=t, source=[arg1], target=[answer],
                             ternary=[arg2], is_q=True)
            q_str = f"Is {arg1} in the {arg2}?\t{answer}"
        elif  world.ent_map.get(arg1) == "object" and world.ent_map.get(arg2) == "location":
            q_ev = DECEvent(kind=QuestionType.YES_NO_OBJ_LOC,
                             timestep=t, source=[arg1], target=[answer],
                             ternary=[arg2], is_q=True)
            q_str = f"Is the {arg1} in the {arg2}?\t{answer}"
        else:
            raise ValueError(f"Unknown relation type: {prop}")
    
    else:
        raise ValueError(f"Unknown relation type: {prop}")
    
    
    
    return q_ev, q_str
        
    

def outputs_to_answers(outs: List[float]) -> List[str]:
    return [label_map.get(o) for o in outs]

def convert_props_to_questions(world, props, outs, verbose_props: bool = True, start_t: int = None):
    """ 
    Convert propositions to list of natural language question/answer pairs
    """
    
    q_events = []
    q_sents = []
    
    if not verbose_props:
        untokenized_props = [world.idx2prop[p] for p in props]
    else:
        untokenized_props = props
    
    for i, (prop, out) in enumerate(zip(untokenized_props, outs)):
        t = start_t+i if start_t else i+1
        q_ev, q_str = prop_to_q(world, prop, out, t)
        q_events.append(q_ev)
        q_sents.append(q_str)
    
    return q_sents, q_events
        
    
        
        
    
    
    
def decs_from_sst(sst: InstanceSST, world: World) -> List[DECStory]:
    """
    Convert an SST instance to a list of bAbI stories. To limit story size
    we split it into one story for each timestep, where each of the timestep's 
    propositions are converted to an appropriate yes no question/answer pair.

    Parameters
    ----------
    sst : InstanceSST
        DESCRIPTION.

    Returns
    -------
    List[DECStory]
        DESCRIPTION.

    """    
    
    babi_stories = []
    cum_sents = []
    
    all_events = sorted([from_world_event(e) for e in world.history.values()], key=lambda x: x[0].timestep)
    # if not len(all_events) == len(sst.texts):
        # print("problem")
    # assert(len(all_events) == len(sst.texts))
    
    numbered_evs = number_events(all_events)
    
    for i, (sent, props, outs) in enumerate(zip(sst.texts, sst.prop_lists, 
                                           sst.outputs)):
        t = i+1
        cum_sents.append(sent)
        question_texts, q_events = convert_props_to_questions(world, props, outs, start_t=t+1)
        
        episode = number_sents(cum_sents + question_texts)
        
        events = numbered_evs[:t] + [[q] for q in q_events]
        
        new_story_uid = shortuuid.ShortUUID().random(length=6)
 
        new_story = DECStory(seed=sst.seed, uid=f"{sst.uid}_{new_story_uid}",
                             babi_story=episode, events=events, task=sst.task)
        babi_stories.append(new_story)
    
    return babi_stories

def transformer_insts_from_sst(sst: InstanceSST, world: World) -> List[TransformerInstance]:
    """
    

    Parameters
    ----------
    sst : InstanceSST
        DESCRIPTION.
    world : World
        DESCRIPTION.

    Returns
    -------
    List[TransformerInstance]
        DESCRIPTION.

    """
    tis = []
    decs = decs_from_sst(sst, world)
    
    for dec in decs:
        dec_tis = dec_story_to_transformer_inputs(dec)
        for i, dec_ti in enumerate(dec_tis):
            dec_ti.id = f"{dec_ti.id}_{i}"  
        tis += dec_tis
        
    
    return tis
    
    
    
if __name__ == '__main__':
    sst_opts = SSTSampleOptions()