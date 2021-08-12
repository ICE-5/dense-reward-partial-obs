import shutil
import pathlib
import pickle
import gym
import copy

from tqdm import tqdm

import numpy as np

from dataloader.utils import FTWindow
from samplers.backward_sampler import BackwardSampler


class NaiveBackwardSampler(BackwardSampler):
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

                        for i in range(self.control_rate):
                            # sample action with control
                            t -= 1

                            action = self.env.action_space.sample()
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

                                action = self.env.action_space.sample()
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

