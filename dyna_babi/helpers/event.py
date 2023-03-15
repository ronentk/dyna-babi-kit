from typing import Optional, List, Dict
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json
from enum import Enum

from ..Actions.Action import ActionType
from ..Questions.Question import QuestionType

class BeliefType(str, Enum):
    KNOWN = "known"
    INDEF = "indef"
    NEGATED = "negated"

@dataclass_json
@dataclass
class QuestionEvent:
    kind: QuestionType
    timestep: int
    source: str
    target: List[str]
    ternary: Optional[str] = None
    supporting_facts: List[int] = field(default_factory=list)
    implicit_facts: List[int] = field(default_factory=list)
    sub_kind: Optional[str] = None
    
@dataclass(order=True)
class Event:
    kind: ActionType = field(compare=False)
    timestep: int = field(compare=True)
    source: str = field(compare=False)
    target: str = field(compare=False)
    location: str = field(compare=False)
    is_coref: bool = field(compare=False, default=False)
    is_conj: bool = field(compare=False, default=False)
    is_all_act: bool = field(compare=False, default=False)
    is_q: bool = field(compare=False, default=False)
    updated_last_known_loc: bool = field(compare=False, default=False)
    gold_belief: BeliefType = field(compare=False, default=BeliefType.KNOWN) 
    ternary: Optional[str] = field(compare=False, default=None)
    loc_event_idxs: Optional[List[int]] = field(compare=False, default=None) # field recording events leading to current location
    
    
    def __post_init__(self):            
        if self.is_coref and self.is_known_loc_change_event:
            self.loc_event_idxs.insert(0, self.timestep-1)
    
    @property
    def supp_idxs(self):
        idxs = []
        if self.is_coref:
            idxs.append(self.timestep-1)
        idxs.append(self.timestep)
        return idxs
    
    @property
    def loc_knowledge_supported(self):
        return len(self.loc_event_idxs) > 0
    
    @property
    def is_known(self):
        return self.gold_belief == BeliefType.KNOWN
    
    @property
    def is_known_loc_change_event(self):
        return self.updated_last_known_loc
        
    
        
def find_all_loc_supports(idxs: List[int], event_dict: Dict[int, Event]):
    """ 
    Find all supporting location events for a set of idxs
    """
    supports = set(idxs)
    new_supps = set()
    evs = [event_dict[idx] for idx in idxs]
    new_supps = new_supps.union(*[set(ev.loc_event_idxs) for ev in evs])
    while new_supps.difference(supports):
        new_idxs = new_supps.difference(supports)
        supports = supports.union(new_idxs)
        evs = [event_dict[idx] for idx in new_idxs]
        new_supps = new_supps.union(*[set(ev.loc_event_idxs) for ev in evs])
    return supports

