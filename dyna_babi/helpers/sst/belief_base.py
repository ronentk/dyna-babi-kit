from typing import Dict, List, Tuple


import numpy as np
from collections import defaultdict

from ..utils import choice_np_rng
from .proposition import ProbProposition

UNKNOWN_PROB = -1.0 # if no belief exists for a proposition (e.g., hasn't yet been mentioned)

def get_props_belief_timeline(diff_props: Dict[int, List[ProbProposition]]) -> Dict[str, List[Tuple[int, float]]]:
    """
    

    Parameters
    ----------
    diff_props : Dict[int, List[ProbProposition]]
        DESCRIPTION.

    Returns
    -------
    Dict[str, List[Tuple[int, float]]]
        DESCRIPTION.

    """
    props_timeline = defaultdict(list)
    
    # iterate over changes in chrono. order
    for t, pprops in sorted(diff_props.items(), key=lambda x: x[0]):
        for p in pprops:
            props_timeline[p.realization].append((t, p.belief))
            
            # TODO change this to use set
            props_timeline[p.realization] = sorted(list(set(props_timeline[p.realization])), key=lambda x: x[0])
    
    return props_timeline

def get_obj_belief_timeline(obj_name: str, diff_props: Dict[int, List[ProbProposition]]) -> List[Tuple[int, ProbProposition]]:
    """
    

    Parameters
    ----------
    diff_props : Dict[int, List[ProbProposition]]
        DESCRIPTION.

    Returns
    -------
    List[Tuple[int, ProbProposition]]
        DESCRIPTION.

    """
    timeline = []
    for t, pprops in diff_props.items():
        for p in pprops:
            if p.args[0] == obj_name and p.name == "at" and p.belief==1.0:
                timeline.append((t,p))
                
    # sort in ascending temporal order
    return sorted(timeline, key=lambda x: x[0])
                
def prop_belief_at_t(prop: str, at_t: int, props_timeline: Dict[str, List[Tuple[int, float]]]) -> float:
        """
        

        Parameters
        ----------
        prop : str
            DESCRIPTION.
        t : int
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        # assuming known props (not those that haven't been mentioned)
        assert(prop in props_timeline)
        
        prob = UNKNOWN_PROB
        event_t = UNKNOWN_PROB
        for t, p in props_timeline[prop]:
            # take last known belief up until target t
            if t <= at_t:
                prob = p
                event_t = t
            else:
                break
        return prob, event_t
        
def props_by_belief_time(at_t: int,
                         props_timeline: Dict[str, List[Tuple[int, float]]],
                         belief_cond_f) -> List[str]:
    """
    

    Parameters
    ----------
    belief : float
        DESCRIPTION.
    at_t : int
        DESCRIPTION.
    props_timeline : Dict[str, List[Tuple[int, float]]]
        DESCRIPTION.

    Returns
    -------
    List[str]
        DESCRIPTION.

    """
    target_props = []
    for p in props_timeline.keys():
        bel, t = prop_belief_at_t(p, at_t, props_timeline)
        if belief_cond_f(bel):
            target_props.append((p, bel, t))
    return target_props
        
        
class BeliefBase:
    def __init__(self, diff_props: Dict[int, List[ProbProposition]], seed: int = None):
        self.diff_props = diff_props
        self.props_timeline = get_props_belief_timeline(diff_props)
        
        # if random seed not provided, sample one at random
        if not seed:
            self.samples_seed = np.random.randint(1, np.iinfo(np.int32).max)
        else:
            self.samples_seed = seed
            
        # rng for sampling propositions from provided story
        self.samples_rng = np.random.RandomState(self.samples_seed)
    


    def sample_props(self, until_t: int, diff_props: bool = True,
                            n_sample_pos: int = 0, n_sample_neg: int = 0, max_per_t: int = 15):
        # skip timesteps where no change occured (means it is question sentence)
        
        target_props = {i: [] for i in range(1, until_t + 1) if len(self.diff_props[i]) > 0}

        for t in sorted(target_props.keys()):
            # prioritize diff propositions - if there are above max_per_t such props, sample max_t.
            # Otherwise, take all of them and draw randomly from the rest to reach max_per_t
            pos_diff_props = []
            neg_diff_props = []
            if diff_props:
                for p in self.diff_props[t]:
                    if p.belief > 0:
                        pos_diff_props.append((p.realization, p.belief, t))
                    elif p.belief == 0.0:
                        neg_diff_props.append((p.realization, p.belief, t))
                    # ignore UNK
            
            if len(pos_diff_props) > max_per_t:
                sampled = choice_np_rng(pos_diff_props, self.samples_rng, 
                                        max_per_t)
                target_props[t] = list(set(sampled)) 
            else:
                other_props = list(set(self.sample_negs(t, n_sample_neg) + self.sample_pos(t, n_sample_pos) + neg_diff_props).difference(set(pos_diff_props)))
                
                # sample up to max_t - |DIFF| propositions
                n_to_sample = min(len(other_props), max_per_t - len(pos_diff_props))
                sampled_other = choice_np_rng(other_props, self.samples_rng, 
                                              n_to_sample)
                target_props[t] = list(set(sampled_other + pos_diff_props))
            
        return target_props
        
    def sample_negs(self, at_t: int, k: int) -> List[ProbProposition]:
        """
        

        Parameters
        ----------
        until_t : int
            DESCRIPTION.
        k : int
            DESCRIPTION.

        Returns
        -------
        List[ProbProposition]
            DESCRIPTION.

        """
        if k == 0:
            return []
        belief_f = lambda x: x == 0.0
        all_negs = props_by_belief_time(at_t=at_t, 
                                        props_timeline=self.props_timeline,
                                        belief_cond_f=belief_f)
        eff_k = min(len(all_negs), k)
        chosen_idxs = self.samples_rng.choice(len(all_negs), eff_k, replace=False)
        return [p for i, p in enumerate(sorted(all_negs)) if i in chosen_idxs]
    
    def sample_pos(self, at_t: int, k: int) -> List[ProbProposition]:
        """
        

        Parameters
        ----------
        until_t : int
            DESCRIPTION.
        k : int
            DESCRIPTION.

        Returns
        -------
        List[ProbProposition]
            DESCRIPTION.

        """
        if k == 0:
            return []
        belief_f = lambda x: x > 0.0
        all_pos = props_by_belief_time(at_t=at_t, 
                                        props_timeline=self.props_timeline,
                                        belief_cond_f=belief_f)
        eff_k = min(len(all_pos), k)
        chosen_idxs = self.samples_rng.choice(len(all_pos), eff_k, replace=False)
        return [p for i, p in enumerate(sorted(all_pos)) if i in chosen_idxs]
    
    
                
        
        
        