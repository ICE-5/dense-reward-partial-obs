import torch
import numpy as np

# from tqdm import tqdm
import random
import pickle
import yaml
import pathlib


from torch.utils.data import Dataset, DataLoader
from dataloader.utils import get_obs_by_code, parse_depth


class PatialObsDataset(Dataset):
    """ partially observable dataset"""

    def __init__(self, config: dict, phase: str = "train"):
        self.config = config
        dataset_dir = (
            config["data_dir"]
            / config["env_name"]
            / config["offset"]
            / config["dataset_name"]
        )

        with open(dataset_dir / "samples" / f"{phase}_sample_codes.pkl", "rb") as f:
            self.sample_codes = pickle.load(f)

    def __len__(self):
        return len(self.sample_codes)

    def __getitem__(self, idx):
        # given idx, randomly sample a sample_code from {phase}_sample_code and return
        # pair_a observations, pair_b observations, goal observations
        a_code = self.sample_codes[idx]
        rollout_name = a_code.split(".")[0]
        b_code = self._generate_sample_from_a(a_code)
        goal_code = f"{rollout_name}.D000.expert"

        return {
            "a_obs": get_obs_by_code(self.config, a_code),
            "b_obs": get_obs_by_code(self.config, b_code),
            "goal_obs": get_obs_by_code(self.config, goal_code),
        }

    def _generate_sample_from_a(self, a_code: str) -> str:
        rollout_name = a_code.split(".")[0]
        rollout_codes = [code for code in self.sample_codes if rollout_name in code]

        depth = parse_depth(a_code)
        in_margin_codes, out_margin_codes = [], []
        margin = self.config["depth_margin"]

        for code in rollout_codes:
            d = parse_depth(code)

            if abs(depth - d) < margin:
                in_margin_codes.append(code)
            else:
                out_margin_codes.append(code)

        if np.random.uniform(0, 1) < 0.5:
            return random.choice(in_margin_codes)
        else:
            return random.choice(out_margin_codes)


if __name__ == "__main__":

    config_path = (
        pathlib.Path(__file__).resolve().parent.parent / "configs" / "debug.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = PatialObsDataset(config, phase="train")
    test_filecode = "R002.D002.expert"
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=8)
    for batch_idx, batch_samples in enumerate(dataloader):
        print("id: ", batch_idx)
        print("samples: ", batch_samples["a_obs"]["ft"].shape)
        import pdb

        pdb.set_trace()
