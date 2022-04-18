import h5py

import pathlib

import numpy as np
from numpy.linalg import norm, solve
from robosuite.environments.base import MujocoEnv
from drpo.samplers.sampler import Sampler


class TemporalSampler(Sampler):
    def __init__(
        self,
        config: dict,
        demo_path: pathlib.Path,
        out_dir: pathlib.Path,
    ) -> None:
        super().__init__(config, demo_path, out_dir)

        # get env action spec
        self.action_low, self.action_high = self.env.action_spec
    
        # init alpha solver
        self.alpha_solver = QuadraticAlphaSolver(
            ref_points=[
                (1.0, np.cos(config["control_angle_goal"] * np.pi / 180)),
                (0.0, np.cos(config["control_angle_start"] * np.pi / 180)),
                (0.5, np.cos(config["control_angle_mid"] * np.pi / 180)),
            ],
        )


    def _sample_step(self, **kwargs):

        demo_grp = self.grp[kwargs["demo_name"]]
        level = kwargs["level"]
        

        for b in range(self.num_branches):
            branch_name = f"branch_{level:03d}_{b:02d}"
            demo_grp.create_group(branch_name)

            
            action = np.random.uniform(self.action_low, self.action_high)

            self.env.step()
            

        pass
        








class QuadraticAlphaSolver:
    def __init__(self, ref_points: list) -> None:
        assert len(ref_points) == 3

        A = np.zeros([3, 3])
        b = np.zeros(3)

        for i, pt in enumerate(ref_points):
            A[i, 0] = pt[0] ** 2
            A[i, 1] = pt[0]
            A[i, 2] = 1
            b[i] = pt[1]

        self.params = solve(A, b)

    def get_alpha(self, x: float or int) -> float:
        return np.dot(self.params, np.array([x ** 2, x, 1]))