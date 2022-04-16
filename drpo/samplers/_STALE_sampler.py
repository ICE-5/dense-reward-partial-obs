import pathlib
import pickle
import random
import gym

from abc import ABC, abstractmethod
from numpy.linalg import norm

class Sampler(ABC):
    def __init__(
        self,
        config: dict,
        env: gym.Env,
        expert_rollouts: list,
        output_dir: pathlib.Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.env = env
        # NOTE: combine expert demo playback and sampling
        self.expert_rollouts = expert_rollouts
        self.output_dir = output_dir

        self.sensors = config["sensor_used_in_sampling"]

        self.control_rate = config["control_rate"]
        self.num_branches = config["num_branches"]
        self.use_history = config["use_history"]

        self.num_steps_per_branch = config["num_steps_per_branch"]

        self.num_rollbacks_per_trial = config["num_rollbacks_per_trial"]
        self.restore_threshold = config["restore_threshold"]
        self.num_rollbacks_final_trial = config["num_rollbacks_final_trial"]

        self.pair_codes = []
        self.train_pair_codes = []
        self.test_pair_codes = []

        self.sample_codes = []
        self.train_sample_codes = []
        self.test_sample_codes = []

        self.num_expert_rollouts = len(expert_rollouts)


    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def _unit_sample(self):
        pass

    # TODO: change from episode-based split to sample-based split
    def split_train_test(self):
        rollout_names = [f"R{x:03d}" for x in range(self.num_expert_rollouts)]
        random.shuffle(rollout_names)
        split = int(self.num_expert_rollouts * self.config["train_test_split"])
        train_rollout_names = rollout_names[:split]

        # moderate sample_codes
        if len(self.sample_codes) == 0:
            try:
                with open(self.output_dir / "sample_codes.pkl", "rb") as f:
                    existing_sample_codes = pickle.load(f)
            except Exception:
                print("sample codes not found")
        else:
            existing_sample_codes = self.sample_codes
        
        # moderate pair_codes
        if len(self.pair_codes) == 0:
            try:
                with open(self.output_dir / "pair_codes.pkl", "rb") as f:
                    existing_pair_codes = pickle.load(f)
            except Exception:
                print("pair codes not found")
        else:
            existing_pair_codes = self.pair_codes


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

        train_pair_codes = [
            pair
            for pair in existing_pair_codes
            if pair[0].split(".")[0] in train_rollout_names
        ]
        test_pair_codes = [
            pair
            for pair in existing_pair_codes
            if pair[0].split(".")[0] not in train_rollout_names
        ]


        pickle.dump(
            train_sample_codes, open(self.output_dir / "train_sample_codes.pkl", "wb"),
        )
        pickle.dump(
            test_sample_codes, open(self.output_dir / "test_sample_codes.pkl", "wb"),
        )
        pickle.dump(
            train_pair_codes, open(self.output_dir / "train_pair_codes.pkl", "wb"),
        )
        pickle.dump(
            test_pair_codes, open(self.output_dir / "test_pair_codes.pkl", "wb"),
        )
    
    def _restore_env_to_timestep(
        self,
        rollout: list,
        timestep: int,
        use_rollback: bool = True,
        success_num_rollbacks=None,):
        if not use_rollback:
            pos, orn = rollout[timestep].pos, rollout[timestep].orn
            self.env.reset(initial_pose=[pos, orn])

        else:
            fold = 1
            exhaust = False
            tried_num_rollbacks = None

            while True:
                if success_num_rollbacks is not None:
                    num_rollbacks = success_num_rollbacks
                else:
                    num_rollbacks = fold * self.num_rollbacks_per_trial

                if (num_rollbacks > self.num_rollbacks_final_trial) or (
                    num_rollbacks > timestep
                ):
                    exhaust = True
                    rollback_timestep = -1
                    num_rollbacks = timestep + 1
                    self.env.reset()

                else:
                    rollback_timestep = timestep - num_rollbacks
                    rollback_expert = rollout[rollback_timestep]
                    pos, orn = rollback_expert.pos, rollback_expert.orn
                    self.env.reset(initial_pose=[pos, orn])

                restored_ft = None

                for i in range(num_rollbacks):
                    t = rollback_timestep + i + 1
                    action = rollout[t].action
                    restored_ft, _, _, info = self.env.step(action)

                # sanity check to see if FT restoration is successful
                expert_ft = rollout[timestep].obs["ft"][0, :]
                delta_ft = norm(restored_ft - expert_ft)

                if delta_ft < self.restore_threshold:
                    self.prGreen(
                        f"SUCCESS | restore F/T timestep {timestep:4d} with {num_rollbacks:4d} steps of rollback"
                    )
                    tried_num_rollbacks = num_rollbacks
                    break

                if exhaust:
                    tried_num_rollbacks = timestep + 1
                    self.prRed(f"FAILURE | restore F/T failed for timestep {timestep:4d}")
                    break

                fold += 1

            return tried_num_rollbacks

    @staticmethod  
    def prGreen(skk):
        print("\033[92m {}\033[00m".format(skk))

    @staticmethod
    def prRed(skk):
        print("\033[91m {}\033[00m".format(skk))

