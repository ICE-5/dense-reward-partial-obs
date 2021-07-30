import shutil
import pathlib
import pickle
import random
import gym

from tqdm import tqdm

import numpy as np
from numpy.linalg import norm, solve

from dataloader.utils import FTWindow


class TemporalVariantBackwardSampler:
    def __init__(
        self,
        config: dict,
        env: gym.Env,
        expert_demo_path: pathlib.Path,
        output_dir: pathlib.Path,
        empty_output_dir: bool = True,
    ) -> None:

        self.config = config
        self.env = env
        self.output_dir = output_dir
        self.num_expert_rollouts = config["num_expert_rollouts"]
        self.num_seeds = config["num_seeds"]
        self.num_samples = config["num_samples"]
        self.control_rate = config["control_rate"]
        self.stop_sampling_threshold = config["stop_sampling_threshold"]
        self.use_history = config["use_history"]

        # create tree dir in output_dir
        if pathlib.Path(output_dir).is_dir():
            if empty_output_dir:
                shutil.rmtree(output_dir)
        else:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=False)

        # load expert demo
        with open(expert_demo_path, "rb") as f:
            self.expert_demo = pickle.load(f)

        # check validity of num_expert_rollouts
        max_num_expert_rollouts = len(self.expert_demo)
        if self.num_expert_rollouts > max_num_expert_rollouts:
            self.num_expert_rollouts = max_num_expert_rollouts

        self.sample_codes = []
        self.train_sample_codes = []
        self.test_sample_codes = []

        # init alpha solver
        self.alpha_solver = QuadraticAlphaSolver(
            ref_points=[
                (0.0, np.cos(config["control_angle_goal"] * np.pi / 180)),
                (1.0, np.cos(config["control_angle_start"] * np.pi / 180)),
                (0.5, np.cos(config["control_angle_mid"] * np.pi / 180)),
            ],
        )

    def sample(self):
        for rollout_idx, rollout in enumerate(
            self.expert_demo[: self.num_expert_rollouts]
        ):
            rollout_name = f"R{rollout_idx:03d}"
            backward_rollout = rollout[::-1]
            timesteps = range(0, len(rollout), self.control_rate)
            rollout_length = len(rollout)

            for depth, timestep in enumerate(
                tqdm(timesteps, "Dataset generation progress: ")
            ):
                depth_name = f"D{depth:03d}"
                depth_path = pathlib.Path(self.output_dir) / rollout_name / depth_name
                depth_path.mkdir(parents=True, exist_ok=False)

                # update and add expert
                expert = backward_rollout[timestep]

                for sensor in self.config["sensor_used"]:
                    globals()[f"{sensor}_layer"] = {}
                    eval(f"{sensor}_layer")["expert"] = expert.obs[sensor]

                expert.rollout_name = rollout_name
                expert.depth_name = depth_name
                self.sample_codes.append(expert.sample_code)

                if depth > 0:
                    prev_timestep = timesteps[depth - 1]
                    prev_expert = backward_rollout[prev_timestep]
                    pos = prev_expert.pos
                    orn = prev_expert.orn

                    for sample in range(self.num_samples):
                        sample_name = f"S{sample:03d}"

                        self.env.reset([pos, orn])

                        # IMPORTANT: create ft window for each sample
                        t = prev_timestep
                        action_sum = expert.action * 0.0
                        ftw = FTWindow(initial_value=prev_expert.obs["ft"] * 0.0,)

                        for i in range(self.control_rate):
                            t += 1
                            alpha = self.alpha_solver.get_alpha(x=(t / rollout_length))
                            history = action_sum if self.use_history else None

                            action = self._sample_action_with_control(
                                cmp_action=backward_rollout[t].action,
                                alpha=alpha,
                                history=history,
                            )
                            action_sum += action
                            ft, _, _, info = self.env.step(action)
                            if i == self.control_rate - 1:
                                # ft sensor observation
                                ftw.insert(ft, 0)
                                # other sensor observation
                                for sensor in self.config["sensor_used"]:
                                    if sensor != "ft":
                                        eval(f"{sensor}_layer")[sample_name] = info[
                                            sensor
                                        ]

                        for i in range(1, self.config["ft_window_size"]):
                            t += 1
                            if t >= rollout_length:
                                break

                            alpha = self.alpha_solver.get_alpha(x=(t / rollout_length))
                            history = action_sum if self.use_history else None

                            action = self._sample_action_with_control(
                                cmp_action=backward_rollout[t].action,
                                alpha=alpha,
                                history=history,
                            )
                            action_sum += action
                            ft, _, _, info = self.env.step(action)
                            ftw.insert(ft, i)

                        if "ft" in self.config["sensor_used"]:
                            ft_layer[sample_name] = ftw.window

                        self.sample_codes.append(
                            f"{rollout_name}.{depth_name}.{sample_name}"
                        )

                for sensor in self.config["sensor_used"]:
                    pickle.dump(
                        eval(f"{sensor}_layer"),
                        open(depth_path / f"{sensor}.pkl", "wb"),
                    )
        pickle.dump(
            self.sample_codes, open(self.output_dir / "sample_codes.pkl", "wb"),
        )

    def split_train_test(self):
        rollout_names = [f"R{x:03d}" for x in range(self.num_expert_rollouts)]
        random.shuffle(rollout_names)
        split = int(self.num_expert_rollouts * self.config["train_test_split"])
        train_rollout_names = rollout_names[:split]

        if len(self.sample_codes) == 0:
            try:
                with open(self.output_dir / "sample_codes.pkl", "rb") as f:
                    existing_sample_codes = pickle.load(f)
            except Exception:
                print("sample codes not found")
        else:
            existing_sample_codes = self.sample_codes

        train_sample_codes = [
            code
            for code in existing_sample_codes
            if code.split(".")[0] in train_rollout_names
        ]
        test_sample_codes = [
            code
            for code in existing_sample_codes
            if code.split(".")[0] not in train_rollout_names
        ]
        pickle.dump(
            train_sample_codes, open(self.output_dir / "train_sample_codes.pkl", "wb"),
        )
        pickle.dump(
            test_sample_codes, open(self.output_dir / "test_sample_codes.pkl", "wb"),
        )

    def _sample_action_with_control(
        self,
        cmp_action: np.ndarray,
        alpha: float = 0.0,
        backward: bool = True,
        history: np.ndarray = None,
    ) -> np.ndarray:
        """Sample action controlled variance

        Args:
            cmp_action (np.ndarray): the action to compare with
            alpha (float, optional): control parameter for cosine similarity. Defaults to 0..
            history (np.ndarray, optional): previously accumulated action, i.e. sum of previous actions. Defaults to None.
        """
        while True:
            action = self.env.action_space.sample()

            # if decided to use accumulated action, i.e. sum of previous actions instead of single action
            if history is not None:
                action += history

            # calculate cosine similarity between actions
            cos_sim = np.dot(action, cmp_action) / (norm(action) * norm(cmp_action))

            if backward:
                # alpha should be negative or 0.
                if cos_sim <= alpha:
                    return action
            else:
                # alpha should be positive or 0.
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

