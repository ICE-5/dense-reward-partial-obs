import pathlib
import numpy as np

from drpo.samplers.sampler import Sampler
from numpy.linalg import norm, solve


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
        self.action_dim = self.env.action_dim

        # init alpha solver
        self.alpha_solver = QuadraticAlphaSolver(
            ref_points=[
                (1.0, np.cos(config["control_angle_goal"] * np.pi / 180)),
                (0.0, np.cos(config["control_angle_start"] * np.pi / 180)),
                (0.5, np.cos(config["control_angle_mid"] * np.pi / 180)),
            ],
        )

    def sample_step(self, **kwargs):

        try:
            demo_name = kwargs["demo_name"]
            demo_states=kwargs["demo_states"]
            demo_actions=kwargs["demo_actions"]
            level = kwargs["level"]
            initial_global_timestep = kwargs["initial_global_timestep"]
        except KeyError:
            print("Missing necessary parameters for sampling.")

        n = len(demo_actions)

        for b in range(self.num_branches):

            sampled_actions = np.zeros([self.num_steps_per_branch, self.action_dim])

            for j in range(self.num_steps_per_branch):
                t = initial_global_timestep + j + 1
                progress = t / n
                alpha = self.alpha_solver.get_alpha(progress)
                sampled_action = self._sample_action_with_control(
                    demo_action=demo_actions[t], alpha=alpha
                )
                sampled_actions[j, :] = sampled_action

            branch_index = level * self.num_branches + b + 1

            self.record_branch(
                demo_name=demo_name,
                demo_states=demo_states,
                demo_actions=demo_actions,
                branch_index=branch_index,
                initial_global_timestep=initial_global_timestep,
                actions=sampled_actions,
            )

    def _sample_action_with_control(
        self, demo_action: np.ndarray, alpha: float = 0.0
    ) -> np.ndarray:
        """Sample action controlled variance

        Args:
            demo_action (np.ndarray): the action to compare with
            alpha (float, optional): control parameter for cosine similarity. Defaults to 0..
        """
        while True:
            sampled_action = np.random.uniform(self.action_low, self.action_high)
            # calculate cosine similarity between actions
            cos_sim = np.dot(sampled_action, demo_action) / (
                norm(sampled_action) * norm(demo_action)
            )

            if cos_sim >= alpha:
                return sampled_action


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
        return np.dot(self.params, np.array([x**2, x, 1]))
