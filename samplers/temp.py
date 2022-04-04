import pathlib

from robosuite.environments.base import MujocoEnv

from envs.envs_launcher import env_creator
from samplers.sampler import Sampler


class Temp(Sampler):
    def __init__(
        self,
        config: dict,
        demo_path: pathlib.Path,
        output_dir: pathlib.Path,
    ) -> None:
        super().__init__(config, demo_path, output_dir)





    def sample():
        pass

