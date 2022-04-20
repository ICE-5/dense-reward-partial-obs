import pickle
import numpy as np
import math
import pathlib
import random
from typing import Tuple

import torch

# import h5py
from h5py._hl.files import File


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

# TODO: remove
def split_test_train(codes_path: pathlib.Path, split_ratio):
    out_dir = codes_path.parents[0]
    with open(codes_path, "rb") as p:
        codes = pickle.load(p)
        n = len(codes)
        s = train_size = int(split_ratio * n)

        random.shuffle(codes)
        train_codes, test_codes = codes[:s], codes[s:]

        pickle.dump(
            train_codes,
            open(out_dir / "train_codes.pkl", "wb"),
        )
        pickle.dump(
            test_codes,
            open(out_dir / "test_codes.pkl", "wb"),
        )


def get_prev_code(code: str) -> str:
    d, b, g, l = _process_code(code)

    if _is_stem(code):
        if g > 1:
            return f"{d}.{b}.{g-1}.{l-1}"
        else:
            raise ValueError("Found code without previous step.")
    else:
        if l > 0:
            return f"{d}.{b}.{g-1}.{l-1}"
        else:
            return f"{d}.0.{g-1}.{g-1}"


def get_ft_window_by_code(data: File, code: str, ft_window_size: int) -> np.ndarray:
    d, b, g, l = _process_code(code)

    branch_ft_arr = data[f"data/{d}/{b}/ft"]

    # if possible to slice within branch
    if l + 1 > ft_window_size:
        return branch_ft_arr[l + 1 - ft_window_size : l + 1, :]
    else:
        # get part_1 from current branch
        part_1 = branch_ft_arr[: l + 1, :]

        if _is_stem(code):
            # get part_2 by padding
            part_2 = np.zeros([ft_window_size - l - 1, 6])
            return np.concatenate([part_2, part_1], axis=0)
        else:
            # get part_2 from stem
            stem_ft_arr = data[f"data/{d}/0/ft"]
            if g + 1 > ft_window_size:
                # no need for part_3 (zero padding)
                part_2 = stem_ft_arr[g + 1 - ft_window_size : g - l, :]
                return np.concatenate([part_2, part_1], axis=0)
            else:
                # need part_3 (zero padding)
                part_2 = stem_ft_arr[: g - l, :]
                part_3 = np.zeros([ft_window_size - g - 1, 6])
                return np.concatenate([part_3, part_2, part_1], axis=0)


def get_obs_by_code(
    data: File, code: str, ft_window_size: int, unsqueeze: bool = False
) -> dict:
    obs = {}
    d, b, _, l = _process_code(code)

    obs["ft"] = get_ft_window_by_code(data, code, ft_window_size)
    obs["action"] = data[f"data/{d}/{b}/action"][l]
    obs["proprio"] = data[f"data/{d}/{b}/proprio"][l]
    obs["image"] = data[f"data/{d}/{b}/image"][l]

    return _process_obs(obs=obs, unsqueeze=unsqueeze)


def get_demo_endpoint_code(codes: list, endpoint_type="init") -> str:
    """Assume all codes are generated from the same demo

    Args:
        codes (list): sample codes
        endpoint_type (str, optional): eiter "init" or "goal". Defaults to "init".

    Raises:
        ValueError: when other endpoint_type is provided

    Returns:
        str: endpoint code
    """
    if endpoint_type == "init":
        d, _, _, _ = codes[0].split(".")
        return f"{d}.0.0.0"
    elif endpoint_type == "goal":
        return max(
            codes, key=lambda x: int(x.split(".")[2]) if x.split(".")[1] == "0" else 0
        )
    else:
        raise ValueError("Invalid endpoint type provided.")


def to_cuda(data_dict: dict):
    if data_dict is None:
        return None
    return {k: v.cuda() for (k, v) in data_dict.items()}


def _is_stem(code: str) -> bool:
    _, b, _, _ = _process_code(code)
    return int(b) == 0


def _process_code(code: str) -> Tuple[str, int, int, int]:
    d, b, g, l = code.split(".")
    b, g, l = int(b), int(g), int(l)
    return (d, b, g, l)


def _process_obs(obs: dict, unsqueeze: bool = False) -> dict:
    processed_obs = {}

    # TODO: is this transpose necessary?
    processed_obs["ft"] = obs["ft"].T  # [ft_window_size, 6] -> [6, ft_window_size]
    processed_obs["action"] = obs["action"]
    processed_obs["proprio"] = obs["proprio"]
    processed_obs["image"] = obs["image"]
    # processed_obs["image"] = obs["image"].transpose((2, 0, 1))  # [128, 128, 3] -> [3, 128, 128]

    # convert to torch tensor
    processed_obs = {k: torch.Tensor(v).double() for (k, v) in processed_obs.items()}

    # unsqueeze for inference
    if unsqueeze:
        processed_obs = {
            k: torch.unsqueeze(v, dim=0) for (k, v) in processed_obs.items()
        }

    return processed_obs


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
