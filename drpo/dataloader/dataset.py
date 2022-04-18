import h5py
import numpy as np
import pathlib
import pickle
import torch
import yaml
from typing import *


from torch.utils.data import Dataset, DataLoader
from drpo.dataloader.utils import *

# TODO: add split test train, loading samples, change name
# class DRPODataset(Dataset):
#     """
#     DRPO dataset
#     """

#     def __init__(self, config: dict, phase: str = "train"):
#         self.config = config
#         dataset_dir = (
#             config["data_dir"]
#             / config["env_name"]
#             / config["offset"]
#             / config["dataset_name"]
#         )

#         with open(dataset_dir / f"{phase}_pair_codes.pkl", "rb") as f:
#             self.pair_codes = pickle.load(f)

#     def __len__(self):
#         return len(self.pair_codes)

#     def __getitem__(self, idx):
#         code_curr = self.pair_codes[idx][0]
#         code_next = self.pair_codes[idx][1]

#         return {
#             "obs_curr": get_obs_by_code(self.config, code_curr),
#             "obs_next": get_obs_by_code(self.config, code_next),
#         }

class DRPODataset(Dataset):
    def __init__(self, data_dir, config) -> None:
        super().__init__()
        data_dir = pathlib.Path(data_dir)

        self.data = h5py.File(data_dir / "data.hdf5", "r")
        self.ft_window_size = config["ft_window_size"]
        with open(data_dir / "codes.pkl", "rb") as p:
            self.codes = pickle.load(p)
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, index):
        code_curr = self.codes[index]
        code_prev = get_prev_code(code_curr)

        return {
            "obs_curr": get_obs_by_code(self.data, code_curr, self.ft_window_size),
            "obs_prev": get_obs_by_code(self.data, code_prev, self.ft_window_size)
        }



if __name__ == "__main__":

    config_path = pathlib.Path("drpo/configs/template.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_dir = pathlib.Path("data")

    dataset = DRPODataset(data_dir, config)

    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4)
    for batch_idx, batch_samples in enumerate(dataloader):
        print(f"batch id: {batch_idx}")
        print("samples curr: \n", batch_samples["obs_curr"]["code"])
        print("samples prev: \n", batch_samples["obs_prev"]["code"])
        # print("samples FT: ", batch_samples["obs_curr"]["ft"].shape)
        # print("samples action: ", batch_samples["obs_curr"]["action"].shape)
        import pdb; pdb.set_trace()
