# Learning Dense Reward With Partial Observation (DRPO)

Please see the instructions below for running the code. The code have been tested on MacOS and Ubuntu. 

- [Learning Dense Reward With Partial Observation (DRPO)](#learning-dense-reward-with-partial-observation-drpo)
  - [0. Installation](#0-installation)
    - [0.1 Dependencies (TBA)](#01-dependencies-tba)
    - [0.2 Install the virtual environment ###](#02-install-the-virtual-environment-)
  - [1. Project configuration file](#1-project-configuration-file)
    - [1.1 Dataset name](#11-dataset-name)
  - [2. Get and process expert demo from RD2](#2-get-and-process-expert-demo-from-rd2)
    - [2.1 Run `rollout.py` in RD2](#21-run-rolloutpy-in-rd2)
      - [Example](#example)
    - [2.2 Save output to specific directory](#22-save-output-to-specific-directory)
      - [Example](#example-1)
    - [2.3 Process rollout](#23-process-rollout)
      - [Example](#example-2)
      - [Output](#output)
  - [3. Generate dataset](#3-generate-dataset)
      - [Example](#example-3)
      - [Output](#output-1)
  - [4. Train DRPO network](#4-train-drpo-network)
      - [Example](#example-4)
      - [Output](#output-2)
  - [4. Evaluate DRPO network](#4-evaluate-drpo-network)
      - [Example](#example-5)
      - [Output](#output-3)

## 0. Installation

### 0.1 Dependencies (TBA)
```
python=3.7
tqdm
pytorch
tensorboard
```

--------------------------------------------------------------------

### 0.2 Install the virtual environment ###  

1. Install the conda virtual environment from the `environment.yml` file by running 
   - `conda env create -f environment_mac.yml` in the terminal for MacOS
   - `conda env create -f environment_ubuntu.yml` in the terminal for Ubuntu

2. Activate the conda environment by running `conda activate rd2` in the terminal. 

Note: 1. If you see the error `ResolvePackageNotFound` when you install the environment, 
         please try to upgrade your conda environment by running `conda update --all`.

--------------------------------------------------------------------

## 1. Project configuration file
Settings of every link in DRPO (process rollout, generate dataset, train network, etc.) are merged into one configuration file for simplicity of control. You can check `configs` folder for a template file and a debug file. Below is a brief explanation of parameters.

| argument                      | relation   | option / format                   | example                           | note                                                                      |
| ----------------------------- | ---------- | --------------------------------- | --------------------------------- | ------------------------------------------------------------------------- |
| `env_name`                    | DATASET    | `"lap-joint"`, `"peg-in-hole"`    | `"lap-joint"`                     | name of environment                                                       |
| `offset`                      | DATASET    |                                   | `"0mm"`                           | offset identifier                                                         |
| `dataset_name`                | DATASET    |                                   |                                   |                                                                           |
| `sensor_used_in_sampling`     | DATASET    | `"ft"`, `"depthmap"`              | ["ft", "depthmap"]                | list of sensors used in sampling                                          |
| `use_gpu`                     | ALL        | `True`, `False`                   | `True`                            | whether to use GPU if available                                           |
| `ft_window_size`              | ALL        |                                   | 8                                 | # FT frames to concatenate                                                |
| `left_append`                 | ALL        | `True`, `False`                   | `False`                           | if set to `True`, FT window will left in right out                        |
| `sampler`                     | SAMPLER    | `"TemporalVariantForwardSampler"` | `"TemporalVariantForwardSampler"` | name of sampler                                                           |
| `num_expert_rollouts`         | SAMPLER    |                                   | 2                                 | # expert rollouts to use in generating dataset, must be larger than 1     |
| `num_branches`                | SAMPLER    |                                   | 5                                 | # branches to sample at each tree depth                                   |
| `num_steps_per_branch`        | SAMPLER    |                                   | 15                                | # steps per each branch                                                   |
| `control_rate`                | SAMPLER    |                                   | 5                                 | # steps between depths                                                    |
| `train_test_split`            | SAMPLER    |                                   | 0.5                               | ratio between training set and test set                                   |
| `num_rollbacks_per_trial`     | FT_RESTORE |                                   | 3                                 | NORMALLY NO NEED TO CHANGE                                                |
| `restore_threshold`           | FT_RESTORE |                                   | 0.000005                          | NORMALLY NO NEED TO CHANGE                                                |
| `num_rollbacks_final_trial`   | FT_RESTORE |                                   | 15                                | NORMALLY NO NEED TO CHANGE                                                |
| `use_history`                 | TV_SAMPLER | `True`, `False`                   | `False`                           | if set to `True`, use accumulated expert action to control sampled action |
| `control_angle_goal`          | TV_SAMPLER |                                   | 15                                | angle to control at goal position                                         |
| `control_angle_start`         | TV_SAMPLER |                                   | 45                                | angle to control at start position                                        |
| `control_angle_mid`           | TV_SAMPLER |                                   | 85                                | angle to control in the middle                                            |
| `architecture`                | NETWORK    | `1`, `2`                          | 1                                 | corresponds to which architecture to use                                  |
| `ft_network_type`             | NETWORK    | `"MLP"`, `"LSTM"`                 | `"MLP"`                           | FT encoder / decoder network type                                         |
| `z_dim`                       | NETWORK    |                                   | 64                                | size of latent embedding                                                  |
| `sensor_used_in_model`        | NETWORK    |                                   | ["ft", "depthmap"]                | DON'T CHANGE FOR DELTA VERSION                                            |
| `initialize_weights`          | TRAIN      | `True`, `False`                   | `True`                            | whether initialize network parameters                                     |
| `batch_size`                  | TRAIN      |                                   | 64                                | batch size                                                                |
| `lr`                          | TRAIN      |                                   | 0.0003                            | learning rate                                                             |
| `weight_decay`                | TRAIN      |                                   | 0.000005                          | weight decay                                                              |
| `num_iters`                   | TRAIN      |                                   | 1000000                           | # max training iterations                                                 |
| `log_freq`                    | TRAIN      |                                   | 200                               | logging frequency (# iterations)                                          |
| `save_freq`                   | TRAIN      |                                   | 200                               | checkpoint frequency (# iterations)                                       |
| `reconstruction_lambda`       | ARC_1      |                                   | 0.2                               | weight of reconstruction loss in architecture #1 loss calculation         |
| `temporal_enforcement_lambda` | ARC_1      |                                   | 10.0                              | weight of temporal enforcement loss in architecture #1 loss calculation   |



### 1.1 Dataset name
Since different trials may need different dataset, it's better to name the dataset with identifiable parameters. In general, it goes with the following format. Example can be found in given config files.
```
<ENV_CODE>_<SIM>_<OFFSET>_<SAMPLER>_<SENSORS_<FT_WINDOW_SIZE>_<NUM_EXPERT_ROLLOUTS>_<NUM_BRANCHS>_<CONTROL_RATE>
```
| variable                | option / format               | example    | note                                                                            |
| ----------------------- | ----------------------------- | ---------- | ------------------------------------------------------------------------------- |
| `ENV_CODE`              | `LAP`, `PEG`                  | `LAP`      | abbrev of lap-joint and peg-in-hole                                             |
| `SIM`                   | `PYBULLET`, `PYATK`           | `PYBULLET` | simulator                                                                       |
| `OFFSET`                | -                             | -          | same as `offset` in config                                                      |
| `SAMPLER`               | `TVFS`, `NFS` , `TVBS`, `NBS` | `TVFS`     | abbrev. of sampler, e.g., `TVFS` is short for temporal variant forward sampling |
| `SENSORS`               | `FD`, `D`, `F`                | `FD`       | `FD` for FT+depthmap, `F` for FT only, `D` for depthmap only                    |
| `<FT_WINDOW_SIZE>`      | `WS<X>`                       | `WS8`      | see `ft_window_size` in config for `<X>`                                        |
| `<NUM_EXPERT_ROLLOUTS>` | `NR<X>`                       | `NR2`      | see `num_expert_rollouts` in config for`<X>`                                    |
| `<NUM_BRANCHS>`         | `NB<X>`                       | `NB5`      | see `num_branches` in config for`<X>`                                           |
| `<CONTROL_RATE>`        | `CR<X>`                       | `CR5`      | see `control_rate` in config for`<X>`                                           |





## 2. Get and process expert demo from RD2
### 2.1 Run `rollout.py` in RD2

Please use the following argument setting when running `rollout.py` in RD2.

| argument       | setting / format     | example      |
| -------------- | -------------------- | ------------ |
| `--out`        | `<ROLLOUT_NAME>.pkl` | `expert.pkl` |
| `--save-info`  | `True`               |              |
| `--use-shelve` | `False`              |              |
| `--no-render`  | `False`              |              |

#### Example
```bash
python rollout.py 'rd2_pybullet_checkpoints/checkpoint_21/checkpoint-21' --out expert_raw.pkl
```

### 2.2 Save output to specific directory

* After rollout, save the pickle file into the following directory in DRPO for future processing.
  ```bash
  /dense-reward-partial-obs/data/<ENV_NAME>/<OFFSET>/rd2/<ROLLOUT_NAME>.pkl
  ```
* Also create a `.csv` with indices of successful rollouts in the same directory
  ```bash
  /dense-reward-partial-obs/data/<ENV_NAME>/<OFFSET>/rd2/<ROLLOUT_NAME>.csv
  ```
* You can close RD2 project after this procedure.

#### Example
* `lap-joint` environment with `0mm` offset
  ```bash
  /data/lap-joint/0mm/rd2/expert.pkl
  /data/lap-joint/0mm/rd2/expert.csv
  ```
* `peg-in-hole` environment with `2mm` offset, rollout name is `expert_fd`
  ```bash
  /data/lap-joint/2mm/rd2/expert_fd.pkl
  /data/lap-joint/2mm/rd2/expert_fd.csv
  ```

### 2.3 Process rollout

Run `process_rollouts.py` in project folder with the following argument setting.

| argument                       | required | setting / format                     | example              |
| ------------------------------ | -------- | ------------------------------------ | -------------------- |
| `-c` / `--config`              | `True`   | path of config file                  | `configs/debug.yaml` |
| `-n` / `--expert-rollout-name` | `True`   | name used in RD2 `rollout.py` output | `expert`             |

#### Example
```bash
python process_rollouts.py -c configs/debug.yaml -n expert
```

#### Output
If successfully run through, you should find processed rollout pickle file in the prompted folder. For example, with lap-joint env, 0mm offset, expert rollout name "expert", there will be two pickle files outputted.

| file                                                 | note                                                                                                                                                             |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/lap-joint/0mm/rd2/processed_expert_8.pkl`      | containing a list of processed expert rollouts, length of list equal to `num_expert_rollouts` in config, the `8` in this example corresponds to `ft_window_size` |
| `data/lap-joint/0mm/rd2/processed_expert_8_best.pkl` | containing a single processed expert rollout, it has minimal rollout length among all successful expert rollouts                                                 |



## 3. Generate dataset

Run `generate_dataset.py` in project folder with the following argument setting.

| argument                       | required | setting / format                                                                                 | example              |
| ------------------------------ | -------- | ------------------------------------------------------------------------------------------------ | -------------------- |
| `-c` / `--config`              | `True`   | path of config file                                                                              | `configs/debug.yaml` |
| `-n` / `--expert-rollout-name` | `True`   | name used in RD2 `rollout.py` output                                                             | `expert`             |
| `--split-only`                 | `False`  | if set to `True`, will not sample again but only perform train-test split among existing dataset | `False`              |

#### Example
```bash
python generate_dataset.py -c configs/debug.yaml -n expert
```

#### Output
If successfully run through, you should find generated dataset in the prompted folder. The folder will be named with `dataset_name` in config.


## 4. Train DRPO network

Run `train.py` in project folder with the following argument setting.

| argument              | required | setting / format                                                   | example                         |
| --------------------- | -------- | ------------------------------------------------------------------ | ------------------------------- |
| `-c` / `--config`     | `True`   | path of config file                                                | `configs/debug.yaml`            |
| `--model-params-path` | `False`  | if want to resume training, just provide the model params .pt file | `checkpoints/<MODEL-ID>/400.pt` |

#### Example
```bash
python train.py -c configs/debug.yaml
```

#### Output
There will be two folders automatically created once the training starts. `checkpoints` stores model params backup, and `logs` store `tensorboard` and `.csv` logs for result monitoring.

> `<MODEL-ID>` is a time identifier of training.
```
checkpoints/<MODEL-ID>/
logs/<MODEL-ID>/
```


## 4. Evaluate DRPO network

Run `eval.py` in project folder with the following argument setting.

| argument                       | required | setting / format                                                                                   | example                         |
| ------------------------------ | -------- | -------------------------------------------------------------------------------------------------- | ------------------------------- |
| `-c` / `--config`              | `True`   | path of config file                                                                                | `configs/debug.yaml`            |
| `-n` / `--expert-rollout-name` | `True`   | name used in RD2 `rollout.py` output                                                               | `expert`                        |
| `-m` / `--model-params-path`   | `True`   | checkpoint model params to evaluate                                                                | `checkpoints/<MODEL-ID>/400.pt` |
| `--use-delta`                  | `False`  | if set to `True`, use delta z from FT to infer the reward, otherwise use z from depthmap to infer. | `False`                         |

#### Example
```bash
python train.py -c configs/debug.yaml
```

#### Output
There will be prompted readings of dense reward and distance reward. Also, plots of reward over each trajectoru can be found in `media/<MODEL-ID>/vis_reward`

> `<MODEL-ID>` is a time identifier of training.
```
checkpoints/<MODEL-ID>/
logs/<MODEL-ID>/
```



