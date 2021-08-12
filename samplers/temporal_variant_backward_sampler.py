import shutil
import pathlib
import pickle
import gym
import copy

from tqdm import tqdm

import numpy as np
from numpy.linalg import norm, solve

from dataloader.utils import FTWindow
from samplers.backward_sampler import BackwardSampler


class TemporalVariantBackwardSampler(BackwardSampler):
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

    def sample(self):
        # empty output directory before sampling
        shutil.rmtree(self.output_dir)

        for rollout_idx, rollout in enumerate(self.expert_rollouts):

            rollout_name = f"R{rollout_idx:03d}"
            rollout_length = len(rollout)

            # this makes sure the we are sampling backward originating from the goal state
            timesteps = [
                rollout_length - 1 - i
                for i in range(1, rollout_length, self.control_rate)
            ]
            if 0 not in timesteps:
                timesteps.append(0)

            for depth, timestep in enumerate(
                tqdm(timesteps, f"Expert rollout #{rollout_idx} sampling progress: ")
            ):
                depth_name = f"D{depth:03d}"
                depth_path = pathlib.Path(self.output_dir) / rollout_name / depth_name
                depth_path.mkdir(parents=True, exist_ok=False)

                # create container layer for each sensor and add expert obs
                expert = rollout[timestep]
                for sensor in self.config["sensor_used_in_sampling"]:
                    globals()[f"{sensor}_layer"] = {}
                    eval(f"{sensor}_layer")["expert"] = expert.obs[sensor]

                expert.rollout_name = rollout_name
                expert.depth_name = depth_name
                self.sample_codes.append(expert.sample_code)

                # samples at depth i originates from expert sample from depth (i - 1)
                if depth > 0:
                    prev_timestep = timesteps[depth - 1]

                    for sample in range(self.num_samples):
                        sample_name = f"S{sample:03d}"

                        # restore env state to timestep of previous depth
                        if "ft" in self.config["sensor_used_in_sampling"]:
                            use_rollback = True
                            ftw = FTWindow(
                                initial_value=np.zeros(
                                    [self.config["ft_window_size"], 6]
                                ),
                            )
                        else:
                            use_rollback = False

                        if sample == 0:
                            success_num_rollbacks = self._restore_env_to_timestep(
                                rollout=rollout,
                                timestep=prev_timestep,
                                use_rollback=use_rollback,
                            )
                        else:
                            self._restore_env_to_timestep(
                                rollout=rollout,
                                timestep=prev_timestep,
                                use_rollback=use_rollback,
                                success_num_rollbacks=success_num_rollbacks,
                            )

                        t = prev_timestep
                        cmp_action = rollout[t].action * 0.0

                        for i in range(self.control_rate):
                            # sample action with control
                            t -= 1
                            progress = t / rollout_length
                            alpha = self.alpha_solver.get_alpha(x=progress)

                            # decide what action to compare
                            if self.use_history:
                                cmp_action += rollout[t].action
                            else:
                                cmp_action = rollout[t].action

                            action = self._sample_action_with_control(
                                cmp_action=cmp_action, alpha=alpha,
                            )

                            ft, _, _, info = self.env.step(action)

                            if i == self.control_rate - 1:
                                # ft sensor observation
                                if "ft" in self.config["sensor_used_in_sampling"]:
                                    ftw.insert(ft, 0)
                                # other sensor observation
                                for sensor in self.config["sensor_used_in_sampling"]:
                                    if sensor != "ft":
                                        eval(f"{sensor}_layer")[sample_name] = info[
                                            sensor
                                        ]

                        # continue sampling to fill the rest of the FTWindow
                        if "ft" in self.config["sensor_used_in_sampling"]:
                            for i in range(1, self.config["ft_window_size"]):
                                t -= 1
                                if t <= 0:
                                    break

                                progress = t / rollout_length
                                alpha = self.alpha_solver.get_alpha(x=progress)

                                if self.use_history:
                                    cmp_action += rollout[t].action
                                else:
                                    cmp_action = rollout[t].action

                                action = self._sample_action_with_control(
                                    cmp_action=cmp_action, alpha=alpha,
                                )
                                ft, _, _, info = self.env.step(action)
                                ftw.insert(ft, i)

                            ft_layer[sample_name] = copy.deepcopy(ftw.window)

                        self.sample_codes.append(
                            f"{rollout_name}.{depth_name}.{sample_name}"
                        )

                for sensor in self.config["sensor_used_in_sampling"]:
                    pickle.dump(
                        eval(f"{sensor}_layer"),
                        open(depth_path / f"{sensor}.pkl", "wb"),
                    )
        pickle.dump(
            self.sample_codes, open(self.output_dir / "sample_codes.pkl", "wb"),
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

            if cos_sim <= alpha:
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
