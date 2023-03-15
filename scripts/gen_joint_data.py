""" 
Script to generate concatenations of babi tasks.

   
Usage:
    gen_joint_data.py TASKS [--babi_dir=<babi_dir> --out_dir=<out_dir>]
    

Options:
   -h --help     Show this screen.
   TASKS         Comma separated task numbers (e.g., [1,20])
   --babi_dir=<babi_dir>    Directory babi data is located at [default: data]
   --out_dir=<out_dir>      Output directory for resulting data. [default: data/concat_bAbI]

"""

from docopt import docopt
from pathlib import Path
import sys
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))

if __name__ == "__main__":
    
    arguments = docopt(__doc__, version='0.1')
    
    rel_babi_path = arguments.get('--babi_dir')
    rel_out_dir = arguments.get('--out_dir')
    babi_data_dir = ROOT / rel_babi_path / "tasks_1-20_v1-2" / "en-valid-10k"
    
    # get tasks to concat
    tasks = sorted([int(x.strip()) for x in arguments.get('TASKS').split(',')])
    # tasks = [str(x) for x in tasks]

    # create output dir
    dir_name = "concat-" + "_".join([str(x) for x in tasks])
    out_dir = ROOT / rel_out_dir / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_files = {
        "train": out_dir / f"{dir_name}_train.txt",
        "valid": out_dir / f"{dir_name}_valid.txt",
        "test": out_dir / f"{dir_name}_test.txt"
        }
    
    # initialize output files
    for out_file in out_files.values():
        out_file.open(mode="w")
    
    
    # save the files here first to write in order
    task_files = []
    for task_file in babi_data_dir.iterdir():
        task, split = task_file.stem.split('_')
        task_num = int(task[2:])
        if task_num in tasks:
            task_files.append((task_num, split, task_file))
    
    task_files.sort(key=lambda x: x[0])
    
    
    # write task data
    for task_num, split, task_file in task_files:
        task_data = task_file.read_text()
        out_file = out_files[split]
        print(f"Writing {task_num} {split} to {out_file}...")
        with out_file.open("a") as f:
                f.write(task_data)
                
    print("Done!")
        
        