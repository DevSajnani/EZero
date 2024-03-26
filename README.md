Code base for training sparse autoencoders on the EfficientZero RL model. The code has been modified to maintain compatibility with SAEs.  <br />
This repo is still under construction 

## Environment
We recommend using a linux distribution and ``conda`` for managing packages.  <br />
First install libgl1-mesa-glx using ```sudo apt-get install libgl1-mesa-glx```  and check if the `libGL.so.1` file exists using  ```find /usr -name "libGL.so.1"```
<br />
After that use pip to install requirements from ``requirements.txt`` and build external cython packages using 
```
cd EZero/core/ctree
bash make.sh
```
Some users might face an error regarding the installation of ROMs when running the original EZero code to train/test an agent, in which case we recommend manually installing them using 
```
AutoROM --accept-license --install-dir {CONDAEnv}/lib/{PYTHON}/site-packages/atari_py/atari_roms
```

## EfficientZero Usage
To run the original EfficientZero model ``cd EZero`` and <br />
* Train: `python main.py --env MsPacmanNoFrameskip-v4 --case atari --opr train --amp_type torch_amp --num_gpus 1 --num_cpus 4 --cpu_actor 1 --gpu_actor 1 --force`
* Test: `python main.py --env MsPacmanNoFrameskip-v4 --case atari --opr test --amp_type torch_amp --num_gpus 1 --load_model --model_path model/model.p ` 

The `model.p` file located under `EZero/model` directory offers good performance, eliminating the need for users to train their own model each time. Users also require GPUs for training new models or sparse autoencoders, but we provide a few of those. We refer users to the original EfficientZero repo (linked under acknowledgements) for more specific advice in the absence of GPUs. 

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


## Sparse Autoencoder Usage
We've only trained and analyzed SAEs on Ms Pacman, but we welcome users to try training some on other environments. Additionally, our SAEs are trained for the final layer of the projection network however this section outlines how SAEs for other layers can be trained. 

### Saving model activations 
To save activations from the final layer for the projection head just add the `--save_activations` argument to test command.
```
cd EZero
python main.py --env MsPacmanNoFrameskip-v4 --case atari --opr test --amp_type torch_amp --num_gpus 1 --load_model --model_path model/model.p --save_activations
```

### Training SAEs


### Feature Visualization using SAEs 
```
cd EZero
python test_neurons.py --env MsPacmanNoFrameskip-v4 --model_path ./model/model.p --sae_paths ./sae/test_sae.p ./sae/test_sae2.p ./sae/test_sae3.p --sae_layers p6 p6 p6 --features 1 5 42 --random_features 2
```

|Required Arguments | Description|
|:-------------|:-------------|
| `--env`                             |Name of the gym environment|

|Other Arguments | Description|
|:-------------|:-------------|
| `--model_path`                           |Path to model
| `--sae_paths`                            |Path to SAE
| `--sae_layers`                           |Layer visualized by the SAE
| `--save_neurons`                         |Neuron attribution if no SAE
| `--device`                               |CPU or Cuda
| `--features`                             |Specific features to attribute
| `--random_features`                      |Number of random features to attribute
| `--feature_source`                       |Encoder or Decoder

In the sample command, three SAEs are used to visualize features from the projection layer 6. The specific features being visualized are 1, 5, and 42, with 2 additional random features being visualized. 

## Contact
If you have any questions please contact sajnanidev@berkeley.edu

## Acknowledgement

```
@inproceedings{ye2021mastering,
  title={Mastering Atari Games with Limited Data},
  author={Weirui Ye, and Shaohuai Liu, and Thanard Kurutach, and Pieter Abbeel, and Yang Gao},
  booktitle={NeurIPS},
  year={2021}
} 
```
[Mastering Atari Games with Limited Data Codebase](https://github.com/YeWR/EfficientZero) <br />
[AI-Safety-Foundation Sparse Autoencoder library](https://github.com/ai-safety-foundation/sparse_autoencoder)

