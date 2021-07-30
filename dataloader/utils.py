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

    @property
    def rollout(self) -> int or None:
        try:
            return float(self.rollout_name[1:])
        except Exception:
            return None

    @property
    def depth(self) -> float or None:
        try:
            return float(self.depth_name[1:])
        except Exception:
            return None


class FTWindow:
    def __init__(self, initial_value: np.ndarray) -> None:
        self.window = initial_value

    def update(self, new_ft: np.ndarray or list) -> None:
        try:
            self.window[1:, :] = self.window[:-1, :]
            self.window[0, :] = new_ft

        except ValueError:
            print("invalid update, check dimension of new_ft")

    def insert(self, new_ft: np.ndarray or list, index: int) -> None:
        try:
            self.window[index, :] = new_ft

        except IndexError:
            print("invalid index for insertion")
        except ValueError:
            print("invalid update, check dimension of new_ft")


def get_sample_by_code(config: dict, sample_code: str) -> np.ndarray:
    sample = Sample()
    sample.sample_code = sample_code

    assert sample.rollout_name is not None
    assert sample.depth_name is not None
    assert sample.sample_name is not None

    data_dir = pathlib.Path(config["data_dir"])
    env_name = config["env_name"]
    samples_dir = data_dir / env_name / "samples"

    for sensor in config["sensor_used"]:
        with open(
            samples_dir / sample.rollout_name / sample.depth_name / f"{sensor}.pkl",
            "rb",
        ) as f:
            tmp = pickle.load(f)

        t = tmp[sample.sample_name]

        if sensor == "ft":
            t = t.T
            if config["left_append"]:
                sample.obs[sensor] = torch.Tensor(t).double()
            else:
                sample.obs[sensor] = torch.flip(torch.Tensor(t), dims=[1,]).double()
        elif "map" in sensor:
            t = np.expand_dims(t, axis=2)
            t = t.transpose((2, 0, 1))
            sample.obs[sensor] = torch.Tensor(t).double()
        elif "img" in sensor:
            t = t.transpose((2, 0, 1))
            sample.obs[sensor] = torch.Tensor(t).double()
        else:
            sample.obs[sensor] = torch.Tensor(t).double()

    # add depth into obs for computing comparison loss
    assert sample.depth is not None
    sample.obs["depth"] = sample.depth

    return sample.obs


def parse_depth(sample_code: str) -> int:
    return int(sample_code.split(".")[1][1:])
