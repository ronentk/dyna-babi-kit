from typing import List, Dict, Optional, Union, Set

import shortuuid
import uuid
from collections import Counter
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json
from ..helpers.event import Event, QuestionEvent
from dataclass_type_validator import dataclass_type_validator
from ..helpers.utils import replace_supp_fact_idx, replace_sent_idx
        
# Hacky, replace with automatic version
ENT_TYPES =  {'bathroom': 'L',
 'bedroom': 'L',
 'garden': 'L',
 'hallway': 'L',
 'kitchen': 'L',
 'office': 'L',
 'cinema': 'L',
 'park': 'L',
 'school': 'L',
 'apple': 'O', 'football': 'O', 'milk': 'O',
 'Daniel': 'P',
 'Mary': 'P',
 'John': 'P',
 'Sandra': 'P',
 'Fred': 'P',
 'Bill': 'P',
 'Jeff': 'P',
 'Julie': 'P'
 }

@dataclass_json       
@dataclass
class DECEvent:
    """
    Represents an event in structured format, to facilitate conversion to Discrete
    Event Calculus (DEC). 
    
    Questions are also represented as events, and are distinguished by is_q flag == True.
    For questions, the `target` field contains the answer tokens.
    
    Attributes
    ----------
    timestep : int
        Time (1-indexed) of event.
    kind : str
        Event/question type.
    source : List[str]
        Source entities.
    target : List[str]
        Target entities.
    ternary : Optional[List[str]]
        Optional auxiliary entities. For example, in the give event, ternary is the object.
    is_coref : bool
        Flag indicating if this event contains a co-reference. Co-references always refer to the
        `source` argument, and the alias is determined via `DECStory.coref_map`.
    is_conj : bool
        Flag indicating if this event is a conjunction ("Mary and John went to the park."). 
        Only possible currently for `move` events. In these cases, there will be two 
        seperate events with the same time step and is_conj == True.
    is_all_act : bool
        Flag indicating if this event contains the all quantifier ("Mary dropped all her possesions."). 
        Only possible currently for `drop` and `give` events. In these cases, there will be n 
        seperate events (for each of n objects) with the same time step and is_all_act == True.
    is_q : bool
        Flag indicating if this event is a question.
    supporting_facts : Optional[List[int]]
        If question event, contains indices of sentences constituting supporting facts for the answer.
    
        
        
    """
    timestep: int
    kind: str
    source: List[str]
    target: Union[List[List[str]], List[str]]
    ternary: Optional[List[str]] = field(default_factory=list) # for give(p1,p2,obj) events this will be obj name 
    is_coref: bool = False
    is_conj: bool = False
    is_q: bool = False
    is_all_act: bool = False
    gold_belief: Optional[str] = "known" # known, indef, or negated
    supporting_facts: Optional[List[int]] = field(default_factory=list)
    implicit_facts: Optional[List[int]] = field(default_factory=list)
    chosen_q: bool = False
    sub_kind: Optional[str] = None
    
    
    def __post_init__(self):
        # ensure all arguments in list form
        if self.ternary == None:
            self.ternary = []
        if not type(self.supporting_facts) is list:
            self.supporting_facts = list(self.supporting_facts)
            
        self.supporting_facts = sorted(self.supporting_facts)
            
        
        # self.gold_belief = self.gold_belief.value if self.gold_belief else "known"
            
        dataclass_type_validator(self)
    
    def to_str(self, names_template: Dict = None) -> str:
        """
        Assuming source and targets are lists of arguments for coref and conj

        Parameters
        ----------
        names_template : Dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        str
            DESCRIPTION.

        """
        names_template = names_template if names_template else {}
        
        # change "john is in the garden" type events to move for purposes of linearization
        # as semantically equivalanet to move.
        # TODO maybe change in DEC as well?
        if self.kind == "negate" and self.gold_belief == "known":
            kind = "move"
        else:
            kind = self.kind
        
        sources_str = ",".join([names_template.get(x, x) for x in self.source])
        if self.is_conj:
            sources_str = f"CONJ({sources_str})"
        if self.is_coref:
            sources_str = f"COREF({sources_str})"
        
        all_args_strs = [sources_str]
        
        if not self.is_q:
            targets_str = ",".join([names_template.get(x, x) for x in self.target])
            if len(self.target) > 1:
                targets_str = f"({targets_str})"
            all_args_strs.append(targets_str)
            
        
        
        
        if self.ternary:
            ternary_str = ",".join([names_template.get(x, x) for x in self.ternary])
            if len(self.ternary) > 1:
                ternary_str = f"({ternary_str})"
            
            all_args_strs += [ternary_str]
        
        all_arg_str = ",".join(all_args_strs)
        final_rep = f"{kind}({all_arg_str})"
        return final_rep
            

def create_templatized_name_map(dec_events: List[DECEvent]):
    """ 
    For provided list of events, return dict mapping each name to a templatized version:
        All entity names converted to {ent_type}_{ent_appearance_order} where 
        `ent_appearance_order`is calculated for each type separately.
        Assuming only one event per timestep (that conj, compound have been unified)
    """        
    ent_type_counter = Counter()
    name_map = {}
    # iterate over events in order of appearance
    for dec_ev in sorted(dec_events, key=lambda x: x.timestep):
        if dec_ev.is_q:
            # ignore target entities if question
            names = sorted(dec_ev.source) + sorted(dec_ev.ternary)
        else:
            names = sorted(dec_ev.source) + sorted(dec_ev.target) + sorted(dec_ev.ternary)
        for name in names:
            if not name in name_map:
                ent_type = ENT_TYPES.get(name)
                ent_type_counter[ent_type] += 1
                templated_name = f"{ent_type}_{ent_type_counter[ent_type]}"
                name_map[name] = templated_name
    return name_map
        
        
        
def unify_timestep_evs(dec_events: List[DECEvent], 
                         names_template: Dict = None) -> str:
        """
        

        Parameters
        ----------
        dec_events : List[DECEvent]
            DESCRIPTION.
        names_template : Dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        str
            DESCRIPTION.

        """
        if len(dec_events) > 1:
            if dec_events[0].is_conj:
                # compound or conj
                compound_event_dict = dec_events[0].to_dict()
                for dec_event in dec_events[1:]:
                    compound_event_dict["source"].append(dec_event.source[0])
            
                final_dec_event = DECEvent.from_dict(compound_event_dict)
            elif dec_events[0].is_all_act:
                # drop or give all action
                compound_event_dict = dec_events[0].to_dict()
                for dec_event in dec_events[1:]:
                    if dec_events[0].kind == "give":
                        compound_event_dict["ternary"].append(dec_event.ternary[0])
                    elif dec_events[0].kind == "drop":
                        compound_event_dict["target"].append(dec_event.target[0])
                    
                final_dec_event = DECEvent.from_dict(compound_event_dict)
            else:
                raise NotImplementedError(f"Unknown event: {dec_events[0]}")
                
        else:
            final_dec_event = dec_events[0]
            
        return final_dec_event   


        
def linearize_events(dec_events: List[DECEvent], templatize_names: bool = True) -> str:
    """
    

    Parameters
    ----------
    names_template : Dict, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    str
        DESCRIPTION.

    """
    
    if templatize_names:
        templatized_names = create_templatized_name_map(dec_events)
    all_evs_str = ""
    for ev in sorted(dec_events, key=lambda x: x.timestep):
        t_evs_str = ev.to_str(names_template=templatized_names)
        sep = "#" if not ev.is_q else "?"
        all_evs_str += (sep + t_evs_str)
    return all_evs_str




@dataclass_json       
@dataclass
class DECStory:
    """
    Attributes
    ----------
    seed : int
        Generation seed, for reproduceability
    events : List[List[DECEvent]]
        Sequence of events, at each time step multiple concurrent events may occur 
    coref_map : Dict[str, List[str]]
        Map between names and possible references e.g., {"Mary": ["she"]} 
    babi_story : List[str]
        Story in bAbI format, as list of sentences
    uid : str
        Unique id of this story
    ie_answers: Dict[int, List[str]]
        dictionary mapping a question index to the inference engine's answer
    ie_s_facts: Dict[int, List[set]]
        dictionary mapping a question index to the inference engine's supporting facts
        supporting facts = sets of sentences-indices from which the answer was inferred
    """
    seed: int
    events: List[List[DECEvent]]
    babi_story: List[str]
    coref_map: Optional[Dict[str, List[str]]] = field(default_factory=dict)
    ie_answers: Optional[Dict[int, Union[List[str],List[List[str]],None]]] = field(default_factory=dict)
    ie_s_facts: Optional[Dict[int, Union[List[List[int]], List[Set[int]], None]]] = field(default_factory=dict)
    uid: str = ""
    task: str = ""
    
    
    def __post_init__(self):
        
        # add uuid if no value provided
        self.uid = shortuuid.uuid(name=str(uuid.uuid4())) if self.uid == "" else self.uid
        self.seed = int(self.seed)
        
        # validate all arguments
        dataclass_type_validator(self)
    
    def story_sent_idxs(self, until_t: int = None, include_qs: bool = False):
        """ 
        Return all story (non-question) events up until timestep `until_t` (including).
        """
        if not until_t:
            until_t = len(self.events)
        idxs = [ev[0].timestep for ev in self.events if (not ev[0].is_q or include_qs) and ev[0].timestep <= until_t]
        return idxs
    
    def story_text(self, until_t: int = None, include_qs: bool = False) -> str:
        """ 
        Return text of story (not including questions) up until timestep 
        `until_t` (including).
        """
        idxs = self.story_sent_idxs(until_t, include_qs)
        sents = []
        for idx in idxs:
            sents.append(self.babi_story[idx-1])
        return "".join(sents)
    
    def print_story(self, until_t: int = None, include_qs: bool = True):
        print(self.story_text(until_t=until_t, include_qs=include_qs))
    
    def question_sent_idxs(self) -> List[int]:
        """ 
        Return timesteps of all questions for this story.
        """
        idxs = [ev[0].timestep for ev in self.events if ev[0].is_q]
        return idxs
    
    def q_sig_str(self, q_timestep: int, templatize_names: bool = True) -> str:
        """ 
        Return string signiture for question + supporting facts
        """
        q_ev = self.ev_by_timestep(q_timestep)[0]
        assert(q_ev.is_q), "Event at t={q_timestep} not question! {q_ev}"
        all_idxs = q_ev.supporting_facts + [q_ev.timestep]
        return self.linearize_timesteps(all_idxs, templatize_names=templatize_names)
    
    def get_supp_comp(self, supp_idxs: List[int]) -> Set[str]:
        """[summary]

        Args:
            supp_idxs (List[int]): [description]

        Returns:
            List[str]: [description]
        """

        unified_ev_list = [unify_timestep_evs(ts) for ts in self.ev_by_timesteps(supp_idxs)]
        support_comp = set()
        for supp_ev in unified_ev_list:
            support_comp.add(supp_ev.kind)
            if supp_ev.is_coref: 
                support_comp.add("co_ref")
            if supp_ev.is_conj: 
                support_comp.add("conj")
        return support_comp
        
    def get_all_supp_comps(self) -> Dict:
        """[summary]

        Returns:
            Dict: [description]
        """
        supp_comps = {}
        for q_idx in self.question_sent_idxs():
            q_ev = self.ev_by_timestep(q_idx)[0]
            supp_comps[q_idx] = self.get_supp_comp(q_ev.supporting_facts)
        return supp_comps
        
    
    def ev_by_timestep(self, t: int) -> DECEvent:
        return self.events[t-1]
    
    def ev_by_timesteps(self, times: List[int]) -> List[DECEvent]:
        evs = [self.ev_by_timestep(t) for t in sorted(times)]
        return evs
    
    
    def linearize_timesteps(self, time_steps: List[int], templatize_names: bool = True) -> str:
        """
        

        Parameters
        ----------
        names_template : Dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        str
            DESCRIPTION.

        """
        all_evs = []
        all_evs_str = ""
        for t in time_steps:
            ev = unify_timestep_evs(self.ev_by_timestep(t))
            all_evs.append(ev)
        
        all_evs_str = linearize_events(all_evs, templatize_names=templatize_names)
        
        return all_evs_str
            
    
    
        

    
def normalize_args(event: Event):
    ev_dict = asdict(event)
    ev_dict["source"] = [event.source] if not type(event.source) is list else event.source
    ev_dict["target"] = [event.target] if not type(event.target) is list else event.target
    if not event.ternary:
        ev_dict["ternary"] = [] 
    else:
        if not type(event.ternary) is list:
            ev_dict["ternary"] = [event.ternary]
        else:
            ev_dict["ternary"] = event.ternary
    
    
    
    return ev_dict
    
def from_world_event(event: Event) -> List[DECEvent]:
    """
    Convert internal world event to Discrete Event Calculus (DEC) format.
    """
    ev_dict = normalize_args(event)
    
    # stringify enum, set gold belief to "known" if non specified 
    ev_dict["gold_belief"] = event.gold_belief.value if event.gold_belief else "known"
    
    dec_event = DECEvent.from_dict(ev_dict)
    dec_events = [dec_event]
    
    # if event is conjunction, split to two events
    if event.is_conj:
        dec_event.source = [dec_event.source[0]]
        
        dec_event_2 = DECEvent.from_dict(normalize_args(event))
        dec_event_2.source = [dec_event_2.source[1]]
        dec_events = [dec_event, dec_event_2]
    
    # if event uses all quantifier, split to n events for each affected object
    if event.is_all_act:
        if event.kind == "drop":
            dec_events = []
            base_event_d = normalize_args(event)
            for target in event.target:
                base_event_d["target"] = [target]
                dec_events.append(DECEvent.from_dict(base_event_d))      
        elif event.kind == "give":
            dec_events = []
            base_event_d = normalize_args(event)
            for ternary in event.ternary:
                base_event_d["ternary"] = [ternary]
                dec_events.append(DECEvent.from_dict(base_event_d))
        else:
            raise NotImplementedError(f"Unknown kind: {event.kind}")
    
    return dec_events

def from_world_q_event(q_event: QuestionEvent) -> List[DECEvent]:
    dec_q_event = DECEvent.from_dict(normalize_args(q_event))
    dec_q_event.is_q = True
    dec_q_events = [dec_q_event] 
    return dec_q_events


def get_q_facts(story: DECStory, q_idx: int) -> List[str]:
    fact_types = []
    q_ev = story.ev_by_timestep(q_idx)[0]
    unified_ev_list = [unify_timestep_evs(ts) for ts in story.ev_by_timesteps(q_ev.supporting_facts)]
    for ev in unified_ev_list:
        if ev.is_all_act:
            ev.kind = f"{ev.kind}_all"
        fact_types.append(ev.kind)
    return fact_types

def repl_q_sent_ans_sf(babi_q_sent: str, answer: List[str], supp_facts: List[int]):
    ans = ",".join(sorted(answer))
    pre_q = babi_q_sent.split("?")[0].split(" ")[0:] # 
    pre_q[-1] = pre_q[-1] + "?"
    q_part = " ".join(pre_q)
    sf_str = " ".join([str(x) for x in supp_facts])
    full_sent = f"{q_part}\t{ans}\t{sf_str}\n"
    return full_sent

def check_dec_answers_consistency(story: DECStory):
    """ 
    Check whether Inference Engine answers match original engine.
    """
    matches = {}
    ie_answers = story.ie_answers if story.ie_answers else {}
    ie_s_facts = story.ie_s_facts if story.ie_s_facts else {}
    for t_q in story.question_sent_idxs():
        if t_q in ie_answers:
            t_q_ie = set()
            
            if isinstance(ie_answers[t_q][0],list):
                # TODO hacky code to address case if answer is list -> was list question
                if ie_answers[t_q][0] == []:
                    # empty list -> convert to "nothing" answer
                    t_q_ie = set(["nothing"])
                else:
                    # or create set of each of the carried objects
                    t_q_ie = set(ie_answers[t_q][0])
            else:
                t_q_ie = set(ie_answers[t_q])
        
        t_q_ie_sf = ie_s_facts.get(t_q, [])
        
        q_ev = story.ev_by_timestep(t_q)[0]
        if isinstance(q_ev.target[0],list):
            if q_ev.target[0] == []:
                # same as above
                t_q_orig = set(["nothing"])
            else:
                t_q_orig = set(q_ev.target[0])
        else:
            t_q_orig = set(q_ev.target)
            
        matches[t_q] = t_q_ie == t_q_orig
        if not matches[t_q] and len(t_q_ie_sf) > 0:
            repl_sent = story.babi_story[t_q - 1]
            q_ev.target = list(t_q_ie)
            
            # get minimal set 
            num_min_s_facts, s_facts = min([(len(s),s) for s in t_q_ie_sf])
            q_ev.supporting_facts = list(s_facts)
            new_sent = repl_q_sent_ans_sf(repl_sent, q_ev.target, q_ev.supporting_facts)
            story.babi_story[t_q - 1] = new_sent
            print(f"Story {story.seed} replace {repl_sent} to {new_sent}")
            
    
    all_match = all(matches.values())
    return all_match
        
    

def filter_keep_timesteps(story: DECStory, keep_idxs: List[int],
            keep_uid: bool = False) -> DECStory:
    """
    Create new DECStory by copying only idxs in `keep_idxs` from `story`.

    Parameters
    ----------
    story : DECStory
        Input DECStory.
    keep_idxs : List[int]
        Sentence idxs to keep.

    Returns
    -------
    DECStory
        new DECStory with only sentences in `keep_idxs`.

    """
    sorted_keep_idxs = sorted(keep_idxs)
    new_old_map = {i+1: t for i,t in enumerate(sorted_keep_idxs)}
    old_new_map = {t: i+1 for i,t in enumerate(sorted_keep_idxs)}
    evs = story.ev_by_timesteps(sorted_keep_idxs)
    new_evs = []
    new_babi_sents = []

    ie_answers={}
    ie_s_facts={}

    old_ie_answers = story.ie_answers if story.ie_answers else {}
    old_ie_s_facts = story.ie_s_facts if story.ie_s_facts else {}

    
    # renumber sentences if needed
    for i, timestep_evs in enumerate(evs):
        old_t = timestep_evs[0].timestep
        s = story.babi_story[old_t-1]
        new_t = old_new_map.get(old_t)
        new_timestep_evs = []
        # renumber event timesteps
        for ev in timestep_evs:
            ev_copy = DECEvent.from_dict(ev.to_dict())
            ev_copy.timestep = new_t
            
            # renumber q supporting facts idxs
            if ev.is_q:
                
                # renumber in babi story
                for i in ev.supporting_facts:
                    s = replace_supp_fact_idx(s, i, old_new_map[i])
                    
                ev_copy.supporting_facts = [old_new_map[old_t] for old_t in ev.supporting_facts]
                ie_answers[new_t] = old_ie_answers.get(old_t, None)
                ie_s_facts[new_t] = old_ie_s_facts.get(old_t)

            new_timestep_evs.append(ev_copy)
        
        # renumber babi sentence
        # use count = 1 to only replace first occurence (sent. idx)
        s = replace_sent_idx(s, old_t, new_t)
        
        # else:
        #     new_timestep_evs = timestep_evs
                    
            
            
            
            
        
        new_evs.append(new_timestep_evs)
        new_babi_sents.append(s)
    
    if not keep_uid:
        new_dec = DECStory(seed=story.seed,
                        events=new_evs,
                        babi_story=new_babi_sents,
                        coref_map=story.coref_map,
                        ie_answers=ie_answers,
                        ie_s_facts=ie_s_facts,
                        task=story.task)
    else:
        new_dec = DECStory(seed=story.seed,
                        events=new_evs,
                        babi_story=new_babi_sents,
                        coref_map=story.coref_map,
                        ie_answers=ie_answers,
                        ie_s_facts=ie_s_facts,
                        task=story.task,
                        uid=story.uid)
    return new_dec
