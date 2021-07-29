# Dense Reward Partial Obs Installation 

Please see the instructions below for running the code. The code have been tested on MacOS and Ubuntu. 

### Dependencies
```
tqdm
PIL
pytorch
```

--------------------------------------------------------------------

### Install the virtual environment ###  

1. Install the conda virtual environment from the `environment.yml` file by running 
   - `conda env create -f environment_mac_py3.7.yml` in the terminal for MacOS
   - `conda env create -f environment_ubuntu.yml` in the terminal for Ubuntu

2. Activate the conda environment by running `conda activate rd2` in the terminal. 

Note: 1. If you see the error `ResolvePackageNotFound` when you install the environment, 
         please try to upgrade your conda environment by running `conda update --all`.

--------------------------------------------------------------------

### Run sampler

Use `test_sampler.py` to run generate sampling. The generated samples will be stored in the `data/<env-name>/samples` folder

```
python test_sampler.py
```

Parameters of the sampler can be specified in `configs/` folder with a YAML file. The parameters are,

- `sampler`: control the type of sampler to use, can be set to `"NaiveBackwardSampler"` or `"TemporalVariantBackwardSampler"`
- `num_expert_rollouts`: control the number of expert rollout to use in sampling, can be set to any large number (e.g.) to indicate using all collected rollouts.
- `num_seeds`: number of seed for sampling at each depth of tree
- `num_samples`: number of samples to generate at each depth
- `control_rate`: interval (in terms of timesteps) to create a depth
- `threshold`: [NAIVE BACKWARD SAMPLER ONLY] threshold to control the end of sampling
- `train_test_split`: ratio of train_set and test_set