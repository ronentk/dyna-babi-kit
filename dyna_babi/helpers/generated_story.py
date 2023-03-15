
from typing import List, Optional
from pathlib import Path
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
import networkx as nx

@dataclass_json
@dataclass
class GeneratedStory:
    """
    Helper class to represent generated story information in structured format
    """
    idx: int # index of story (in [0,N-1] for batch of N stories) 
    sentences: List[str]
    graph: nx.DiGraph() # graph representing full event history of story
    question: str
    answer: str
    seed: int # random seed used to generate stories, for reproduceability
    support_sent_idxs: Optional[List[int]] = field(default_factory=lambda: list())# optionally - list of support sentence indices
    
    @classmethod
    def load(cls, file_path: Path):
        """ 
        Load generated story from file.
        """
        return cls.from_json(file_path.read_text())
        
    def save(self, file_path: Path):
        """ 
        Save generated story to file.
        """
        file_path.write_text(self.to_json())
        
        
    def to_str(self) -> str:
        answer = "yes" if self.answer else "no"
        story_str = ""
        supp_str = " ".join([str(x) for x in self.support_sent_idxs])
        story_str += "\n".join(self.sentences)
        story_str += "\n" + f"{self.question}\t{answer}\t{supp_str}" 
        return story_str