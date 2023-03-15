# README


Dyna-bAbI is a controllable task generator for creating new tasks in the [bAbI](https://arxiv.org/abs/1502.05698) micro-world.

Dyna-babi can be used to create mixtures of the original bAbI tasks, allowing an evaluation of models' compositional generalization abilities.

For the background and motivation, please see the [project page](https://www.notion.so/Dyna-bAbI-project-page-3585f82147734f5a9c6790e0152e8552).

# Quick-start: download pre-generated data

The splits used in the paper can be downloaded as a [single archive](https://drive.google.com/file/d/18j9rEnIaa0C9S1DyWY3eQwQNtTKcdJcu/view?usp=sharing) or from [Weights & Biases](https://wandb.ai/dyna-babi/datasets/artifacts/).

# Advanced: generate tasks from scratch

For a deeper dive into task generation, this section describes how to set up new tasks, and also provides details on how to generate the tasks used in our experiments.

## Task configuration

In dyna-bAbI, our splits are typically comprised of sets of tasks, or you could also think of them as a composite task comprised of multiple sub-tasks. 

We define a single task in dyna-bAbI using a `TaskConfig` configuration file comprised of 4 main components:

- `StoryParameters`: defines generation parameters including the number of stories to generate, the number questions per story, and available concept types (e.g., types of actions, linguistic constructs and questions). See the [documentation](dyna_babi/game_variables_parser.py#L64) for details.
- `FilterConfig`: defines the filters to use for rejection sampling. For example, filters can be configured to pass only stories of certain length, certain question types, and questions with a supporting fact set of particular size and/or composition. Multiple filters can be used for a single task; any generated instance that passes at least one of these will be added to the task. See the [documentation](dyna_babi/story_filter.py#L28) for details.
- `GameVariables`: defines the entity vocabularies; the possible `Person` , `Object` and `Location` entities. If not specified, will default to the sets used in the original bAbI tasks. See the [documentation](dyna_babi/game_variables_parser.py#L10) for details.
- `StoryWriterConfig` (optional): Additional task specific generation parameters can optionally be set here, if you want to override the defaults set in `TaskConfig`. See the [documentation](dyna_babi/story_writer.py#L69) for details. 

So a full dataset is defined by a `TasksConfig` file, which includes a set of `TaskConfig`s as well as parameters controlling output formats. See the [documentation](dyna_babi/tasks_writer.py#L36) for details. 

**new inference engine** - The original inference engine used to calculate the supporting facts set has a number of limitations - it doesn't recover all possible supporting fact sets, and doesn't cover all possible reasoning patterns. Since the release of the paper, we've experimented with a new inference engine that addresses these issues, though it's still experimental so use at your own risk :) To use it set `use_new_engine: True` in the `TasksConfig` file (or per specifc task in the `StoryWriterConfig`).

## Usage

### Install requirements

(Tested using Python 3.8)

From repo root, run:

```
pip install -r requirements.txt
```


### Creating the `mix` and `diverse` datasets

From repo root, run:

```
python generate_tasks.py  --tasks_config=configs/{diverse,mix}_{T7,T12}.json
```

By default, train, dev and test splits will be created from a `TasksConfig` file, where the size of the dev and test sets are 1/10 the size of the train set. Task generation for hasn't yet been parallelized so it can take a while if many filters are applied, such as in the `mix` datasets (several hours for `mix(T12)`).

### Creating the `inject` datasets

The `inject` splits are created using the `solve_babi_tasks.py` script which serves to enriching an existing dataset with specified question types. For each question in the original data, the script adds all possible questions of the types specified in the `solver_config.json` configuration file. 

To create the `inject` data used in the experiments:

- First get the original bAbI data, by running (from repo root): `./get_orig_babi_data.sh`
- Next, run:
    
    ```
    python scripts/solve_babi_tasks.py data/tasks_1-20_v1-2/en-valid-10k/ --task_configs=configs/solver/inject_{T7,T12}.json --trim_over --odir=data/inject_{T7,T12}
    ```
    
    (the `--trim_over` flag only takes stories that are less than 500 tokens long, to fit into the maximum input window size of the T5.)
    

### Creating the `concat` datasets

Run:

```
python scripts/gen_joint_data.py {T7,T12} --babi_dir=data
```

Where T7=`1,2,3,5,11,12,13` and T12=`1,2,3,5,6,7,8,9,10,11,12,13`

---

## Citation

```

@inproceedings{tamari-etal-2022-dyna,
    title = "{D}yna-b{A}b{I}: unlocking b{A}b{I}{'}s potential with dynamic synthetic benchmarking",
    author = "Tamari, Ronen  and
      Richardson, Kyle  and
      Kahlon, Noam  and
      Sar-shalom, Aviad  and
      Liu, Nelson F.  and
      Tsarfaty, Reut  and
      Shahaf, Dafna",
    booktitle = "Proceedings of the 11th Joint Conference on Lexical and Computational Semantics",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.starsem-1.9",
    pages = "101--122",
}
```