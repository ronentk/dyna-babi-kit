from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import numpy as np
from dacite import from_dict, Config

from multiprocessing import Pool

from .game_variables_parser import StoryParameters, GameVariables
from .story_writer import StoryWriter, StoryWriterConfig
from .story_filter import FilterConfig
from .helpers.sst.instance_sst import SSTSampleOptions

logging.basicConfig(level = logging.INFO)


@dataclass_json
@dataclass
class TaskConfig:
    story_params: StoryParameters = field(default_factory=lambda: StoryParameters())
    game_variables: GameVariables = field(default_factory=lambda: GameVariables())
    filter_config: Optional[Union[FilterConfig, List[FilterConfig]]] = None
    sw_config: Optional[StoryWriterConfig] = None
    
    @property
    def task_name(self):
        return self.story_params.name

@dataclass_json
@dataclass
class TasksConfig:
    name: str
    tasks: List[TaskConfig]
    out_dir: str = "default"
    write_tt: bool = False
    write_dec: bool = False
    no_write: bool  = False
    only_dev: bool = False
    combine_files: bool = False
    just_combine: bool = False # don't write 
    save_separate: bool = True
    use_new_engine: bool = False

    @property
    def out_path(self):
        return Path(self.out_dir) / self.name
    
    def __post_init__(self):
        if not self.combine_files and not self.save_separate:
            print("At least `combine_files` or `save_separate` must be True, "
                  "forcing save_separate=True...")
            self.save_seperate = True
    
    def to_pretty_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)
            
    @classmethod
    def from_json_file(cls, task_json_file: str):
        task_json_file = Path(task_json_file)
        tasks_config = json.load(task_json_file.open())
        full_tasks_config = from_dict(data_class=cls, data=tasks_config,
                                      config=Config(check_types=False))
        return full_tasks_config

        
    @classmethod
    def from_jsons(cls, json_data: str):
        tasks_config = json.loads(json_data)
        full_tasks_config = from_dict(data_class=cls, data=tasks_config,
                                      config=Config(check_types=False))
        return full_tasks_config

        
    
        
    
def combine_files(source_files: List[Path], target_file: Path):
    for f in source_files:
        data = f.read_text()
        data += "\n"
        with target_file.open("a+") as f_out:
            f_out.write(data)
    


class TasksWriter:
    def __init__(self, tasks_config: TasksConfig):
        self.tasks_config = tasks_config
        self.story_writers = []

        for n, task_config in enumerate(self.tasks_config.tasks):
            enum_task_name = f"{n+1}_{task_config.task_name}" 
            task_out_dir = tasks_config.out_path / enum_task_name
            if not task_config.sw_config:
                sw_config = StoryWriterConfig(out_dir=str(task_out_dir),
                                              write_tt=tasks_config.write_tt,
                                              write_dec=tasks_config.write_dec,
                                              no_write=tasks_config.no_write,
                                              use_new_engine=tasks_config.use_new_engine,
                                              only_dev=tasks_config.only_dev
                                              )
                task_config.sw_config = sw_config
            else:
                sw_config = task_config.sw_config

            story_writer = StoryWriter(sw_config,
                                    story_parameters=task_config.story_params,
                                    game_variables=task_config.game_variables,
                                    filter_config=task_config.filter_config
                                  )
            self.story_writers.append(story_writer)
    @property
    def no_write(self) -> bool:
        return self.tasks_config.no_write
   
    @property
    def out_dir(self) -> Path:
        return self.tasks_config.out_path
    
    def prepare_out_dir(self):
        
        # erase dir if exists previously
        if self.tasks_config.out_path.exists():
            logging.warn(f"Output dir {self.tasks_config.out_path} already exists, over-writing...")
            shutil.rmtree(self.tasks_config.out_path)
        
        self.tasks_config.out_path.mkdir(parents=True, exist_ok=True)
        
        # save config file
        out_param_file = self.tasks_config.out_path / "tasks_config.json"
        out_param_file.write_text(self.tasks_config.to_pretty_json())
        

    def combine_tasks(self):
        files = defaultdict(list)
        
        # collect babi format task files (.txt)
        for file_path in self.tasks_config.out_path.glob("**/*.txt"):
            if (("train" in file_path.stem) or 
                ("test" in file_path.stem) or 
                ("valid" in file_path.stem)):
                files[file_path.name].append(file_path)
        
        # collect dec/tt format task files if exist (.jsonl)
        for file_path in self.tasks_config.out_path.glob("**/*.jsonl"):
            files[file_path.name].append(file_path)
            
        
        
        # create file
        combined_dir = self.out_dir / "combined"
        combined_dir.mkdir(exist_ok=True, parents=True)
            
        
        # combine files for each split/format
        for fname, format_files in files.items():
            logging.info(f"Combining {len(format_files)} {fname} files: {format_files}")
            dest = combined_dir / fname
            combine_files(source_files=format_files, target_file=dest)
            
        
        # save config file
        out_param_file = combined_dir / "tasks_config.json"
        out_param_file.write_text(self.tasks_config.to_pretty_json())
            
    
        
        return files
    
    def generate(self):

        self.generate_tasks()
    
    def generate_tasks(self):
        
        if not self.no_write and not self.tasks_config.just_combine:
            self.prepare_out_dir()
        
        if not self.tasks_config.just_combine:
            for story_writer in self.story_writers:
                story_writer.write_data()
        else:
            # if just combining - don't write any stories
            logging.info(f"Skipping data generation- just combining existing data at {self.tasks_config.out_path}")
            
        if not self.no_write:
            if self.tasks_config.combine_files:
                self.combine_tasks()
        
            if not self.tasks_config.save_separate:
                for sw in self.story_writers:
                    if Path(sw.config.out_dir).exists():
                        shutil.rmtree(sw.config.out_dir)
                        
    def calc_tasks_stats(self):
        # calculate stats for analytics purposes
        stats = {}
        for sw in self.story_writers:
            name = sw.params.name
            stats[name] = {}
            for split, decs in sw.dec_stories.items():
                all_s_facts = []
                for d in decs:
                    all_s_facts += [len(v) for v in d.ie_s_facts.values()]
                all_s_facts = np.array(all_s_facts)
                stats[name][split] = {
                    "split": split,
                    "avg_supp_facts": np.mean(all_s_facts),
                    "mult_path_qs": np.sum(all_s_facts > 1),
                    "total_qs": len(all_s_facts)
                    }
        return stats
                
                
                
                
                
        
    
    
    @classmethod
    def from_json(cls, task_json_file: str):
        task_json_file = Path(task_json_file)
        full_tasks_config = TasksConfig.from_json_file(task_json_file)
        return cls(tasks_config=full_tasks_config)
            

