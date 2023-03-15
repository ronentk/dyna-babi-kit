from typing import List, Dict, Optional, Union, Tuple

import shortuuid
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json
from ..helpers.event_calc import DECStory, DECEvent, filter_keep_timesteps


def story_q_to_transformer_q(sents: List[str], q_timestep: int):
    q_idx = q_timestep - 1
    assert(len(sents) >= q_timestep)
    story_lines = [line for line in sents[:q_idx] if not "?" in line]
    question, answer = process_question_sent(sents[q_idx])
    problem_input = "%s $question$ %s" %\
                            (' '.join([p for p in story_lines]), question)
    return problem_input, answer
        
    
def process_question_sent(sent: str) -> Tuple[str,str]:
    assert("?" in sent)
    question, answer = sent.split("?")
    answer = ' '.join([i for i in answer.strip().split() if not i.isnumeric()])
    question = "%s?" % question
    return question, answer
    
def process_line(line: str):
    line = line.strip()
    detail = ' '.join(line.split()[1:])
    return detail
    
def strip_sents(dec_story: DECStory) -> List[str]:
    return [process_line(s) for s in dec_story.babi_story] 

@dataclass_json       
@dataclass
class TransformerInstance:
    id: str
    task: str
    seed:int
    q_sub_id: int
    input: str
    output: str
    answerKey: int = -1
    prefix: str = "answer:" 
    question: Dict[str,str] = field(default_factory=dict)
    supporting_facts: List[int] = field(default_factory=list)
    chosen_q: bool = False
    
    def __post_init__(self):
        self.question["stem"] = self.input
        self.id += f"_{self.q_sub_id}"

def dec_story_to_transformer_inputs(dec_story: DECStory, filter_distractors: bool = False) -> List[TransformerInstance]:
    """
    Convert story from DEC format to TransformerInstance

    """
    transformer_inputs = []
    stripped_lines = strip_sents(dec_story)
    for q_timestep in dec_story.question_sent_idxs():
        uid = dec_story.uid
        seed = dec_story.seed
        q_ev = dec_story.ev_by_timestep(q_timestep)[0]
        if filter_distractors:
            # remove all distractor timesteps for the current q
            filtered_dec = filter_keep_timesteps(dec_story, 
                                                 keep_idxs= q_ev.supporting_facts + [q_timestep])
            stripped_lines = strip_sents(filtered_dec)
            
            new_q_timestep = len(q_ev.supporting_facts) + 1 # recalc new question idx
            problem_input, answer = story_q_to_transformer_q(stripped_lines, new_q_timestep)
            supp_facts = filtered_dec.ev_by_timestep(new_q_timestep)[0].supporting_facts
            
        else:
            problem_input, answer = story_q_to_transformer_q(stripped_lines, q_timestep)
            supp_facts = dec_story.ev_by_timestep(q_timestep)[0].supporting_facts
            
        
        transformer_inputs.append(
            TransformerInstance(id=uid,
                                task=dec_story.task,
                                seed=seed,
                                q_sub_id=q_timestep, # use original qid to match dec file
                                input=problem_input,
                                output=answer,
                                supporting_facts=supp_facts,
                                chosen_q=q_ev.chosen_q)
            )
    return transformer_inputs
        
    