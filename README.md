Code base for training SAEs on the EfficientZero RL model. 
Work done as part of the Fall 2023 iteration of the Supervised Program for Alignment Research (SPAR)

## Environments
We recommend using a linux distribution and ``conda`` for managing packages. 
First install libgl1-mesa-glx using ```sudo apt-get install libgl1-mesa-glx``` 
and check if the `libGL.so.1` file exists using 
```find /usr -name "libGL.so.1"```
After that use pip to install requirements from ``requirements.txt`` and build external cython packages using 
```cd EZero/core/ctree
bash make.sh```

### Prerequisites
Before starting training, you need to build the c++/cython style external packages. (GCC version 7.5+ is required.)
```
cd core/ctree
bash make.sh
``` 

### Installation
As for other packages required for this codebase, please run `pip install -r requirements.txt`.

## Usage
### Quick start
* Train: `python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --amp_type torch_amp --num_gpus 1 --num_cpus 10 --cpu_actor 1 --gpu_actor 1 --force`
* Test: `python main.py --env BreakoutNoFrameskip-v4 --case atari --opr test --amp_type torch_amp --num_gpus 1 --load_model --model_path model.p \`
### Bash file
We provide `train.sh` and `test.sh` for training and evaluation.
* Train: 
  * With 4 GPUs (3090): `bash train.sh`
* Test: `bash test.sh`

|Required Arguments | Description|
|:-------------|:-------------|
| `--env`                             |Name of the environment|
| `--case {atari}`                    |It's used for switching between different domains(default: atari)|
| `--opr {train,test}`                |select the operation to be performed|
| `--amp_type {torch_amp,none}`       |use torch amp for acceleration|

|Other Arguments | Description|
|:-------------|:-------------|
| `--force`                           |will rewrite the result directory
| `--num_gpus 4`                      |how many GPUs are available
| `--num_cpus 96`                     |how many CPUs are available
| `--cpu_actor 14`                    |how many cpu workers
| `--gpu_actor 20`                    |how many gpu workers
| `--seed 0`                          |the seed
| `--use_priority`                    |use priority in replay buffer sampling
| `--use_max_priority`                |use the max priority for the newly collectted data
| `--amp_type 'torch_amp'`            |use torch amp for acceleration
| `--info 'EZ-V0'`                    |some tags for you experiments
| `--p_mcts_num 8`                    |set the parallel number of envs in self-play 
| `--revisit_policy_search_rate 0.99` |set the rate of reanalyzing policies
| `--use_root_value`                  |use root values in value targets (require more GPU actors)
| `--render`                          |render in evaluation
| `--save_video`                      |save videos for evaluation

## Contact
If you have any question or want to use the code, please contact sajnanidev@berkeley.edu

## Acknowledgement
```
@inproceedings{ye2021mastering,
  title={Mastering Atari Games with Limited Data},
  author={Weirui Ye, and Shaohuai Liu, and Thanard Kurutach, and Pieter Abbeel, and Yang Gao},
  booktitle={NeurIPS},
  year={2021}
}
```
