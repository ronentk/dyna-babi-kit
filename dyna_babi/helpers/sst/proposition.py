from typing import Tuple
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class ProbProposition:
    args: Tuple[str]
    name: str = "at"
    belief: float = 1.0
    
    
    def __str__(self) -> str:
        args_str = ",".join(self.args)
        return f"({self.name}({args_str}), b={self.belief})"
    
    @property
    def prop(self) -> str:
        args_str = ",".join(self.args)
        return f"({self.name}({args_str})"
    
    @property
    def prop_sent(self) -> str:
        assert(len(self.args) == 2), "Currently only supporting 2 argument propositions"
        subj, obj = self.args
        if self.name == "held":
            return f"{obj} holds {subj}"
        else:
            return f"{subj} {self.name} {obj}"
        
    @property
    def realization(self) -> str:
        return self.prop_sent
        