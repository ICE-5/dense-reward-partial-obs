import shutil
import pathlib
import pickle
import gym
import copy

from tqdm import tqdm

import numpy as np
from numpy.linalg import norm, solve

from dataloader.utils import FTWindow
from samplers.sampler import Sampler


class TemporalVariantForwardSampler(Sampler):
    def __init__(
        self,
        config: dict,
        env: gym.Env,
        expert_rollouts: list,
        output_dir: pathlib.Path,
    ) -> None:
        super().__init__(
            config=config,
            env=env,
            expert_rollouts=expert_rollouts,
            output_dir=output_dir,
        )

        # init alpha solver
        self.alpha_solver = QuadraticAlphaSolver(
            ref_points=[
                (1.0, np.cos(config["control_angle_goal"] * np.pi / 180)),
                (0.0, np.cos(config["control_angle_start"] * np.pi / 180)),
                (0.5, np.cos(config["control_angle_mid"] * np.pi / 180)),
            ],
        )
        self.pair_codes = []

    def sample(self):
        # empty output directory before sampling
        shutil.rmtree(self.output_dir)

        for rollout_idx, rollout in enumerate(self.expert_rollouts):

            rollout_name = f"R{rollout_idx:03d}"
            rollout_length = len(rollout)

            # sampling forward
            sampling_timesteps = [
                i for i in range(0, rollout_length, self.control_rate)
            ]

            expert_branch_name = "expert"
            expert_branch_path = (
                pathlib.Path(self.output_dir) / rollout_name / expert_branch_name
            )
            expert_branch_path.mkdir(parents=True, exist_ok=False)
            expert_branch_ft, expert_branch_depthmap = {}, {}

            for t in range(rollout_length - 1):
                curr_t_name = f"T{t:03d}"
                next_t_name = f"T{(t+1):03d}"
                self.pair_codes.append(
                    (
                        f"{rollout_name}.{expert_branch_name}.{curr_t_name}",
                        f"{rollout_name}.{expert_branch_name}.{next_t_name}",
                    )
                )
                self.sample_codes.append(
                    f"{rollout_name}.{expert_branch_name}.{curr_t_name}"
                )

                if t == rollout_length - 2:
                    expert_branch_ft[next_t_name] = rollout[t + 1].obs["ft"]
                    expert_branch_depthmap[next_t_name] = rollout[t + 1].obs["depthmap"]
                    self.sample_codes.append(
                        f"{rollout_name}.{expert_branch_name}.{next_t_name}"
                    )
                expert_branch_ft[curr_t_name] = rollout[t].obs["ft"]
                expert_branch_depthmap[curr_t_name] = rollout[t].obs["depthmap"]

                if t in sampling_timesteps:

                    depth = sampling_timesteps.index(t)
                    for b in range(self.num_branches):
                        branch_name = f"D{depth:03d}-B{b:03d}"
                        branch_path = (
                            pathlib.Path(self.output_dir) / rollout_name / branch_name
                        )
                        branch_path.mkdir(parents=True, exist_ok=False)
                        branch_ft, branch_depthmap = {}, {}

                        # restore state to t
                        if b == 0:
                            success_num_rollbacks = self._restore_env_to_timestep(
                                rollout=rollout,
                                timestep=t,
                                use_rollback=True,
                                success_num_rollbacks=None,
                            )
                        else:
                            self._restore_env_to_timestep(
                                rollout=rollout,
                                timestep=t,
                                use_rollback=True,
                                success_num_rollbacks=success_num_rollbacks,
                            )

                        inbranch_t = t
                        cmp_action = rollout[t].action * 0.0

                        ftw = FTWindow(
                            initial_value=copy.deepcopy(rollout[t].obs["ft"])
                        )

                        for i in range(self.num_steps_per_branch):
                            inbranch_t += 1
                            progress = inbranch_t / rollout_length
                            alpha = self.alpha_solver.get_alpha(progress)

                            if self.use_history:
                                cmp_action += rollout[inbranch_t].action
                            else:
                                cmp_action = rollout[inbranch_t].action

                            action = self._sample_action_with_control(
                                cmp_action=cmp_action, alpha=alpha,
                            )
                            ft, _, _, info = self.env.step(action)
                            ftw.update(ft)
                            branch_ft[inbranch_t] = copy.deepcopy(ftw.window)
                            branch_depthmap[inbranch_t] = info["depthmap"]

                            self.sample_codes.append(
                                f"{rollout_name}.{branch_name}.{curr_t_name}"
                            )

                            if i < self.num_steps_per_branch - 1:
                                curr_t_name = f"T{inbranch_t:03d}"
                                next_t_name = f"T{(inbranch_t+1):03d}"
                                self.pair_codes.append(
                                    (
                                        f"{rollout_name}.{branch_name}.{curr_t_name}",
                                        f"{rollout_name}.{branch_name}.{next_t_name}",
                                    )
                                )

                            if i == self.num_steps_per_branch - 1:
                                self.sample_codes.append(
                                    f"{rollout_name}.{branch_name}.{next_t_name}"
                                )

                            if inbranch_t >= rollout_length - 1:
                                break

                        pickle.dump(
                            branch_ft, open(branch_path / "ft.pkl", "wb"),
                        )

                        pickle.dump(
                            branch_depthmap, open(branch_path / "depthmap.pkl", "wb"),
                        )
                pickle.dump(
                    expert_branch_ft, open(expert_branch_path / "ft.pkl", "wb"),
                )

                pickle.dump(
                    expert_branch_depthmap,
                    open(expert_branch_path / "depthmap.pkl", "wb"),
                )

        pickle.dump(
            self.pair_codes, open(self.output_dir / "pair_codes.pkl", "wb"),
        )

    def _sample_action_with_control(
        self, cmp_action: np.ndarray, alpha: float = 0.0
    ) -> np.ndarray:
        """Sample action controlled variance

        Args:
            cmp_action (np.ndarray): the action to compare with
            alpha (float, optional): control parameter for cosine similarity. Defaults to 0..
        """
        while True:
            action = self.env.action_space.sample()

            # calculate cosine similarity between actions
            cos_sim = np.dot(action, cmp_action) / (norm(action) * norm(cmp_action))

            if cos_sim >= alpha:
                return action


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
