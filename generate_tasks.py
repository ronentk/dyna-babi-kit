

import argparse
import logging
from dyna_babi.tasks_writer import TasksWriter, TasksConfig


logging.basicConfig(level = logging.INFO)


def parse_args():
    
    description = """
    Create multiple tasks set in the bAbI world. Customizable generation controlled via TasksConfig configuration file, 
    where different generation parameters can be specified for each task in json format.
    """

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--tasks_config",
        help="Path to tasks config file.",
        type=str,
        default="_configs/sample_tasks_config.json"
    )

    parser.add_argument(
        "--out_dir",
        help="name of the output dir where the data will be written.",
        type=str
    )
    
    parser.add_argument(
        "--write_dec",
        help="Flag controlling whether to write dec graphs to output dir. (default: False)",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--write_tt",
        help="Flag controlling whether to write output in transformer tools format. (default: False)",
        action='store_true',
        default=False
    )

    
    parser.add_argument(
        "--combine_files",
        help="Combine all tasks into one file per split/format. (default: False)",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--erase_separate",
        help="Erase separate folders per task. (default: False)",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--use_new_engine",
        help="Use new inference engine. (default: False)",
        action='store_true',
        default=False
    )
    
    parser.add_argument(
        "--just_combine",
        help="Don't write any data, just combine files in specified out dir. (default: False)",
        action='store_true',
        default=False
    )
    
    
    

    args = parser.parse_args()
    tasks_config = TasksConfig.from_json_file(args.tasks_config)
    if args.out_dir:
        tasks_config.out_dir = args.out_dir
    if args.write_tt:
        tasks_config.write_tt = True
        
    if args.write_dec:
        tasks_config.write_dec = True
        
    

    if args.use_new_engine:
        tasks_config.use_new_engine = True
        
    if args.just_combine:
        tasks_config.just_combine = True
        tasks_config.combine_files = True
        
    
    if args.erase_separate:
        tasks_config.save_separate = False
        if not args.combine_files:
            logging.warn("At least `combine_files=True` or `erase_separate=False` , "
                  "forcing erase_separate=False...")
            tasks_config.save_separate = True

    if args.combine_files:
        tasks_config.combine_files = True

    return tasks_config




if __name__ == "__main__":

    tasks_config = parse_args()

    tasks_writer = TasksWriter(tasks_config)
    tasks_writer.generate()
