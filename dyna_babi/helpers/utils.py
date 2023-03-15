from typing import Set, List
import numpy as np
import numpy.random as random
import re

RANDOM_SEED = -1
NUM_RE = re.compile(r"(\d+)") # to capture supporting facts idxs
IDX_RE = re.compile(r"^(\d+)(?:\s+)")



    

def replace_sent_idx(babi_sent: str, old_idx: int, new_idx: str):
    # make sure arg is a string
    repl = str(new_idx) + " " # add space between line number and sentence
    return IDX_RE.sub(repl, babi_sent)

def replace_supp_fact_idx(babi_sent: str, old_idx: int, new_idx: int):
    res = babi_sent
    repl_re = re.compile(r"(\s+)(" + str(old_idx) + ")(\s+|$)")
    found = repl_re.findall(babi_sent)
    assert(len(found) <= 1), f"len(found) > 1! found = {found}, babi_sent={babi_sent}, old/new_idx = {old_idx}/{new_idx}"
    if found:
        l,repl,r = found[0]
        repl_str = "".join([l,str(new_idx),r])
        res = repl_re.sub(repl_str, babi_sent)
        
    return res

def reformat_lines(lines: List[str]):
    """ 
    """
    new_lines = []
    for line in lines:
        line = line.strip()
        new_lines.append(line + "\n")
        
    return new_lines


def renumber_story(lines: List[str]):
    
    old_to_new_lines = {}
    
    
    # collect old numbering
    for i, l in enumerate(lines):
        t = i+1
        old_idx = IDX_RE.findall(l)
        if old_idx:
            old_to_new_lines[int(old_idx[0])] = t
    
    new_lines = []
    for l in lines:
        if not "?" in l:
            new_lines.append(l)
            continue
        removed_idx = replace_sent_idx(l, 0,"")
        supp_facts = extract_supp_facts(removed_idx)
        for s in supp_facts:
            # add special symbol to prevent replacing twice
            rep_str = str(old_to_new_lines[s]) + "#"
            l = replace_supp_fact_idx(l,s,rep_str)
        
        # remove special symbol
        l = l.replace("#", "")
        
        new_lines.append(l)
            
    
    # remove old line numbers
    replaced = [replace_sent_idx(s, 0,"").lstrip() for s in new_lines]
    
    # replace with new and return 
    return [f"{i+1} {s}" for i,s in enumerate(replaced)]
        
def supp_facts_str(supp_fact_idxs):
    # ensure sorted in ascending order
    supp_fact_idxs = sorted(list(supp_fact_idxs))
    idxs = " ".join([str(s) for s in supp_fact_idxs])
    return f"\t{idxs}"

def extract_supp_facts(s):
    supp_facts = [int(x) for x in NUM_RE.findall(s)]
    return supp_facts

def seed(s, checked_seeds: Set[int] = None):
    """ 
    Seed random number generator for reproduceability, return seed value.
    If checked seeds supplied, only use seed not in checked seeds
    """
    if s == RANDOM_SEED:
        rand_seed = np.random.randint(1, np.iinfo(np.int32).max)
        if checked_seeds:
            while rand_seed in checked_seeds:
                rand_seed = np.random.randint(1, np.iinfo(np.int32).max)
    else:
        rand_seed = s
    
    np.random.seed(rand_seed)
    return rand_seed

def tuple_to_str(t):
    return tuple(str(i) for i in t)

def choices_np(items, weights):
    """ 
    Convert between `random.choices()` to `numpy.random.choice()`.
    """
    weights = np.array(weights)
    p = weights / weights.sum()
    item = random.choice(items, p=p)
    return item

def choice_np(items):
    """ 
    Replace functionality of calling random choice in numpy
    """
    # items may be a set, so first convert to list
    items_l = list(items)
    if not isinstance(items_l[0], tuple):
        sorted_items = sorted([(str(i), i) for i in items_l])
    else:
        # if list items are a tuple , we first need to call a str() on individual elements
        # and not on the tuple as a whole
        sorted_items = sorted([(tuple_to_str(i), i) for i in items_l])
    item_name, item = sorted_items[random.choice(len(sorted_items), 1)[0]]
    return item

def choice_np_rng(items, rng, k, replace: bool = False):
    """ 
    Replace functionality of calling random choice in numpy, with predefined rng
    """
    # items may be a set, so first convert to list
    items_l = list(sorted(items))
    if k >= len(items_l):
        return items_l

    eff_k = min(len(items_l), k)
    chosen_idxs = rng.choice(len(items_l), eff_k, replace=replace)
    return [item for i, item in enumerate(items_l) if i in chosen_idxs]

def sorted_item_set(item_set):
    # to iterate over set items in the same order for reproduceability
    sorted_items = sorted([(str(i), i) for i in list(item_set)])
    sorted_item_names, sorted_items = zip(*sorted_items)
    return sorted_items

def story_to_t5_format(sentences):
    sents = []
    for line in sentences:
        line = line.strip()
        sents.append(line[2:].strip())
    return " ".join(sents)
