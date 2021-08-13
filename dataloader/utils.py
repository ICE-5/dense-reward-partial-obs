import pickle
import numpy as np
import pathlib
import torch


class Sample:
    def __init__(
        self,
        sample_name: str or None = None,
        rollout_name: str or None = None,
        depth_name: str or None = None,
        action: np.ndarray or None = None,
        obs: dict = {},
        pos: np.ndarray or None = None,
        orn: np.ndarray or None = None,
        dist_reward: float or None = None,
    ) -> None:
        self.sample_name = sample_name
        self.rollout_name = rollout_name
        self.depth_name = depth_name
        self.action = action
        self.obs = obs
        self.pos = pos
        self.orn = orn
        self.dist_reward = dist_reward

    @property
    def sample_code(self) -> str:
        if self.rollout_name is None or self.depth_name is None:
            raise ValueError(
                "sample_code not available, please update rollout_name and depth_name"
            )
        return f"{self.rollout_name}.{self.depth_name}.{self.sample_name}"

    @sample_code.setter
    def sample_code(self, sample_code: str) -> None:
        # TODO: add safety & sanity check
        self.rollout_name, self.depth_name, self.sample_name = sample_code.split(".")


class FTWindow:
    def __init__(self, initial_value: np.ndarray) -> None:
        self.window = initial_value

    def update(self, new_ft: np.ndarray or list) -> None:
        try:
            self.window[1:, :] = self.window[:-1, :]
            self.window[0, :] = new_ft

        except ValueError:
            print("invalid update, check dimension of new_ft")

    def rollback(self,) -> None:
        self.window[:-1, :] = self.window[1:, :]
        self.window[-1, :] = 0.0

    def insert(self, new_ft: np.ndarray or list, index: int) -> None:
        try:
            self.window[index, :] = new_ft

        except IndexError:
            print("invalid index for insertion")
        except ValueError:
            print("invalid update, check dimension of new_ft")


# TODO: refactor
def get_obs_by_code(config: dict, sample_code: str) -> dict:
    """Retrieve processed observation by sample code

    Args:
        config (dict): configuration
        sample_code (str): identifier string for each sample, e.g. R000.D001.expert (rollout #0, depth #1, expert sample)

    Returns:
        dict: processed observation, key being name of used sensors, also including depth
    """
    try:
        rollout_name, branch_name, timestep_name = sample_code.split(".")
    except Exception:
        print(sample_code)
    timestep = int(timestep_name[1:])

    dataset_dir = (
        config["data_dir"]
        / config["env_name"]
        / config["offset"]
        / config["dataset_name"]
    )

    raw_obs = {}
    for sensor in config["sensor_used_in_model"]:
        with open(
            dataset_dir / rollout_name / branch_name / f"{sensor}.pkl", "rb",
        ) as f:
            raw_obs[sensor] = pickle.load(f)[timestep_name]

    obs = process_raw_sample_obs(config, raw_obs)
    # obs["depth"] = float(timestep)

    return obs


def process_raw_sample_obs(
    config: dict, raw_obs: dict, unsqueeze: bool = False
) -> dict:
    """Process raw observations from backward sampling for torch pipeline

    Args:
        config (dict): configuration
        raw_obs (dict): observation from backward sampling

    Returns:
        dict: processed observation ready for torch pipeline
    """
    processed_obs = {}
    for sensor in config["sensor_used_in_model"]:
        t = raw_obs[sensor]

        if "ft" in sensor:
            t = t.T
            if not config["left_append"]:
                t = np.flip(t, axis=1).copy()
        elif "map" in sensor:
            t = np.expand_dims(t, axis=2)
            t = t.transpose((2, 0, 1))
        elif "img" in sensor:
            t = t.transpose((2, 0, 1))
        else:
            pass

        processed_obs[sensor] = torch.Tensor(t).double()

    if unsqueeze:
        return {k: torch.unsqueeze(v, dim=0) for (k, v) in processed_obs.items()}
    else:
        return processed_obs


def parse_depth(sample_code: str) -> int:
    return int(sample_code.split(".")[1][1:])


def to_cuda(data_dict: dict):
    if data_dict is None:
        return None
    return {k: v.cuda() for (k, v) in data_dict.items()}

