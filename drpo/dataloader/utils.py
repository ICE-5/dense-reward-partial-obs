import pickle
import numpy as np
import math
import pathlib
import random
from typing import Tuple

import torch

# import h5py
from h5py._hl.files import File


def split_test_train(codes_path: pathlib.Path, split_ratio: float):
    """Sample-wise split. Potentially mixing multiple demos

    Args:
        codes_path (pathlib.Path): upstream sampled codes
        split_ratio (float): train-test split ratio
    """
    out_dir = codes_path.parents[0]
    with open(codes_path, "rb") as p:
        codes = pickle.load(p)
        n = len(codes)
        s = int(split_ratio * n)

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


# TODO: TBA. for generalization purposes
def split_test_train_2(codes_path: pathlib.Path, split_ratio: float = 0.5):
    """Demo-wise split. E.g., one demo will be used for training, another one will be used for testing.

    Args:
        codes_path (pathlib.Path): upstream sampled codes
        split_ratio (float, optional): train-test split ratio. Defaults to 0.5.
    """
    pass


def get_prev_code(code: str) -> str:
    d, b, g, l = process_code(code)

    if _is_stem(code):
        if g >= 1:
            return f"{d}.{b}.{g-1}.{l-1}"
        else:
            print(code)
            raise ValueError("Found code without previous step.")
    else:
        if l > 0:
            return f"{d}.{b}.{g-1}.{l-1}"
        else:
            return f"{d}.0.{g-1}.{g-1}"


def get_ft_window_by_code(data: File, code: str, ft_window_size: int) -> np.ndarray:
    d, b, g, l = process_code(code)

    branch_ft_arr = data[f"data/{d}/{b}/ft"]

    # if possible to slice within branch
    if l + 1 > ft_window_size:
        return branch_ft_arr[l - ft_window_size + 1 : l + 1, :]
    else:
        # get part_1 from current branch
        part_1 = branch_ft_arr[: l + 1, :]

        if _is_stem(code):
            # get part_2 by padding
            part_2 = np.zeros([ft_window_size - l - 1, 6])
            print(part_1.shape, part_2.shape)
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
    data: File,
    code: str,
    ft_window_size: int,
    use_action_in_delta: bool = True,
    unsqueeze: bool = False,
) -> dict:
    obs = {}
    d, b, _, l = process_code(code)

    if ft_window_size > 1:
        obs["ft"] = get_ft_window_by_code(data, code, ft_window_size)
        assert np.all(np.equal(obs["ft"][-1, :], data[f"data/{d}/{b}/ft"][l]))
    else:
        obs["ft"] = data[f"data/{d}/{b}/ft"][l]

    obs["action"] = data[f"data/{d}/{b}/action"][l]
    obs["proprio"] = data[f"data/{d}/{b}/proprio"][l]
    obs["image"] = data[f"data/{d}/{b}/image"][l]
    obs["depth"] = data[f"data/{d}/{b}/depth"][l]

    # COMMENT OFF: for debug and eval
    obs["code"] = code
    obs["reward"] = data[f"data/{d}/{b}/reward"][l]

    return _process_obs(
        obs=obs, use_action_in_delta=use_action_in_delta, unsqueeze=unsqueeze
    )


def get_demo_endpoint_code(
    codes: list, demo_name: str, endpoint_type: str = "init"
) -> str:
    """Get a demo's endpoint (init or goal) code

    Args:
        codes (list): sample codes
        demo_name (str): name of the demo
        endpoint_type (str, optional): eiter "init" or "goal". Defaults to "init".

    Raises:
        ValueError: when other endpoint_type is provided

    Returns:
        str: endpoint code
    """
    if endpoint_type == "init":
        return f"{demo_name}.0.0.0"
    elif endpoint_type == "goal":
        return max(
            codes,
            key=lambda x: int(x.split(".")[2])
            if (x.split(".")[1] == "0" and x.split(".")[0] == demo_name)
            else 0,
        )
    else:
        raise ValueError("Invalid endpoint type provided.")


def get_demo_codes_by_name(codes: list, demo_name: str) -> list:
    demo_codes = [
        code
        for code in codes
        if (code.split(".")[1] == "0" and code.split(".")[0] == demo_name)
    ]
    return sorted(demo_codes, key=lambda x: int(x.split(".")[2]))


def process_code(code: str) -> Tuple[str, int, int, int]:
    d, b, g, l = code.split(".")
    b, g, l = int(b), int(g), int(l)
    return (d, b, g, l)


def to_cuda(data_dict: dict):
    if data_dict is None:
        return None
    return {k: v.cuda() for (k, v) in data_dict.items()}


def _is_stem(code: str) -> bool:
    _, b, _, _ = process_code(code)
    return int(b) == 0


def _process_obs(obs: dict, use_action_in_delta: True, unsqueeze: bool = False) -> dict:
    processed_obs = {}
    # ft
    if use_action_in_delta:
        processed_obs["ft"] = np.concatenate([obs["ft"].flatten(), obs["action"]])
    else:
        processed_obs["ft"] = obs["ft"]
    # action, [7, ]
    processed_obs["action"] = obs["action"]
    # proprio, [32,]
    processed_obs["proprio"] = obs["proprio"]
    # image, [128, 128, 3] -> [3, 128, 128]
    processed_obs["image"] = obs["image"].transpose((2, 0, 1))
    processed_obs["depth"] = obs["depth"].transpose((2, 0, 1))

    # convert to torch double tensor
    processed_obs = {k: torch.Tensor(v).double() for (k, v) in processed_obs.items()}

    # unsqueeze for inference
    if unsqueeze:
        processed_obs = {
            k: torch.unsqueeze(v, dim=0) for (k, v) in processed_obs.items()
        }

    return processed_obs
