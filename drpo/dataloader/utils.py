import pickle
import numpy as np
import pathlib

# import torch
import copy


# TODO: remove
# class Sample:
#     def __init__(
#         self,
#         action: np.ndarray or None = None,
#         obs: dict = {},
#         pos: np.ndarray or None = None,
#         orn: np.ndarray or None = None,
#         dist_reward: float or None = None,
#     ) -> None:
#         self.action = action
#         self.obs = obs
#         self.pos = pos
#         self.orn = orn
#         # QUESTION: is dist_reward still needed in sample
#         self.dist_reward = dist_reward

# TODO: implement
# BEST: remove to save space
# class Sample:
#     def __init__(self, branch_index, global_timestep, local_timestep, prev=None) -> None:
#         self.branch_index = branch_index
#         self.global_timestep = global_timestep  # sample's index in the stem / universal frame
#         self.local_timestep = local_timestep  # sample's index within the branch frame
#         self.prev = prev


# class FTWindow:
#     def __init__(self, initial_value: np.ndarray) -> None:
#         self.window = copy.deepcopy(initial_value)

#     def update(self, new_ft: np.ndarray or list) -> None:
#         try:
#             self.window[1:, :] = self.window[:-1, :]
#             self.window[0, :] = new_ft

#         except ValueError:
#             print("invalid update, check dimension of new_ft")

#     def rollback(
#         self,
#     ) -> None:
#         self.window[:-1, :] = self.window[1:, :]
#         self.window[-1, :] = 0.0

#     def insert(self, new_ft: np.ndarray or list, index: int) -> None:
#         try:
#             self.window[index, :] = new_ft

#         except IndexError:
#             print("invalid index for insertion")
#         except ValueError:
#             print("invalid update, check dimension of new_ft")


# TODO: refactor
# def get_obs_by_code(config: dict, sample_code: str) -> dict:
#     """Retrieve processed observation by sample code

#     Args:
#         config (dict): configuration
#         sample_code (str): identifier string for each sample, e.g. R000.D001.expert (rollout #0, depth #1, expert sample)

#     Returns:
#         dict: processed observation, key being name of used sensors, also including depth
#     """
#     rollout_name, branch_name, sample_name = sample_code.split(".")

#     dataset_dir = (
#         config["data_dir"]
#         / config["env_name"]
#         / config["offset"]
#         / config["dataset_name"]
#     )

#     raw_obs = {}
#     for sensor in config["sensor_used_in_model"]:
#         with open(
#             dataset_dir / rollout_name / branch_name / f"{sensor}.pkl",
#             "rb",
#         ) as f:
#             raw_obs[sensor] = pickle.load(f)[sample_name]

#     obs = process_raw_sample_obs(config, raw_obs)

#     return obs

def get_obs_by_code(code: str) -> str:
    pass

def get_prev_code(code: str) -> str:
    # branch_index, global_timestep, local_timestep
    b, g, l = [int(i) for i in code.split(".")]

    # if demo / stem
    if b==0:
        if g > 1:
            return f"{b}.{g-1}.{l-1}"
        else:
            return None
    # if sampled branches
    else:
        if l > 0:
            return f"{b}.{g-1}.{l-1}"
        else:
            return f"0.{g-1}.{g-1}"

def split_test_train(codes_path: pathlib.Path, out_dir):
    pass

def get_ft_window(window_size: int):
    pass



# def process_raw_sample_obs(
#     config: dict, raw_obs: dict, unsqueeze: bool = False
# ) -> dict:
#     """Process raw observations from backward sampling for torch pipeline

#     Args:
#         config (dict): configuration
#         raw_obs (dict): observation from backward sampling

#     Returns:
#         dict: processed observation ready for torch pipeline
#     """
#     processed_obs = {}
#     for sensor in config["sensor_used_in_model"]:
#         t = raw_obs[sensor]

#         if "ft" in sensor:
#             t = t.T
#             if not config["left_append"]:
#                 t = np.flip(t, axis=1).copy()
#         elif "map" in sensor:
#             t = np.expand_dims(t, axis=2)
#             t = t.transpose((2, 0, 1))
#         elif "img" in sensor:
#             t = t.transpose((2, 0, 1))
#         else:
#             pass
#         processed_obs[sensor] = torch.Tensor(t).double()

#     if unsqueeze:
#         return {k: torch.unsqueeze(v, dim=0) for (k, v) in processed_obs.items()}
#     else:
#         return processed_obs


def to_cuda(data_dict: dict):
    if data_dict is None:
        return None
    return {k: v.cuda() for (k, v) in data_dict.items()}
