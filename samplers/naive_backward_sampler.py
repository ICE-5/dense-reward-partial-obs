import copy
import shutil
import pathlib
import pickle
import random
import gym
from dataloader.utils import FTWindow

from tqdm import tqdm


class NaiveBackwardSampler:
    def __init__(
        self,
        config: dict,
        env: gym.Env,
        expert_rollouts: list,
        output_dir: pathlib.Path,
    ) -> None:
        self.config = config
        self.env = env
        self.expert_rollouts = expert_rollouts
        self.output_dir = output_dir

        self.num_expert_rollouts = len(self.expert_rollouts)
        self.num_seeds = config["num_seeds"]
        self.num_samples = config["num_samples"]
        self.control_rate = config["control_rate"]

        self.sample_codes = []
        self.train_sample_codes = []
        self.test_sample_codes = []

    def sample(self):
        # empty output directory before sampling
        shutil.rmtree(self.output_dir)
        
        for rollout_idx, rollout in enumerate(
            self.expert_rollouts[: self.num_expert_rollouts]
        ):
            rollout_name = f"R{rollout_idx:03d}"
            backward_rollout = rollout[::-1]
            timesteps = range(0, len(rollout), self.control_rate)

            for depth, timestep in enumerate(
                tqdm(timesteps, f"Expert rollout #{rollout_idx} sampling progress: ")
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

                        ftw = FTWindow(initial_value=prev_expert.obs["ft"] * 0.0,)
                        for i in range(self.control_rate):
                            action = self.env.action_space.sample()
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

                        # fill the remaining slots of ft window (idx=1 ~ ft_window_size-1)
                        for i in range(1, self.config["ft_window_size"]):
                            action = self.env.action_space.sample()
                            ft, _, _, info = self.env.step(action)
                            ftw.insert(ft, i)

                        if "ft" in self.config["sensor_used"]:
                            ft_layer[sample_name] = copy.deepcopy(ftw.window)

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

