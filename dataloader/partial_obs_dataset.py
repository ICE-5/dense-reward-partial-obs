import torch
import numpy as np

# from tqdm import tqdm
import pickle
import yaml
import pathlib


from torch.utils.data import Dataset, DataLoader
from dataloader.utils import *


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

        with open(dataset_dir / f"{phase}_pair_codes.pkl", "rb") as f:
            self.pair_codes = pickle.load(f)

    def __len__(self):
        return len(self.pair_codes)

    def __getitem__(self, idx):
        code_curr = self.pair_codes[idx][0]
        code_next = self.pair_codes[idx][1]

        return {
            "obs_curr": get_obs_by_code(self.config, code_curr),
            "obs_next": get_obs_by_code(self.config, code_next),
        }


if __name__ == "__main__":

    config_path = (
        pathlib.Path(__file__).resolve().parent.parent / "configs" / "debug.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["data_dir"] = pathlib.Path(__file__).resolve().parent.parent / "data"

    dataset = PatialObsDataset(config, phase="train")
    # test_filecode = "R002.D002.expert"
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=8)
    for batch_idx, batch_samples in enumerate(dataloader):
        print("id: ", batch_idx)
        print("samples: ", batch_samples["obs_curr"]["ft"].shape)
        import pdb; pdb.set_trace()
