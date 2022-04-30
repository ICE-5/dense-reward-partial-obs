import h5py
from pathlib import Path
import pickle
import yaml
from typing import *


from torch.utils.data import Dataset, DataLoader
from drpo.dataloader.utils import *


class DRPODataset(Dataset):
    def __init__(self, config: dict, data_path: Path, codes_path: Path) -> None:
        super().__init__()

        self.data = h5py.File(data_path, "r")
        self.ft_window_size = config["ft_window_size"]
        self.use_action_in_delta = config["use_action_in_delta"]
        self.use_object_in_proprio = config["use_object_in_proprio"]
        with open(codes_path, "rb") as p:
            self.codes = pickle.load(p)

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, index: int) -> dict:
        code_curr = self.codes[index]
        code_prev = get_prev_code(code_curr)

        return {
            "obs_curr": get_obs_by_code(
                data=self.data,
                code=code_curr,
                ft_window_size=self.ft_window_size,
                use_action_in_delta=self.use_action_in_delta,
                use_object_in_proprio=self.use_object_in_proprio,
                unsqueeze=False,
            ),
            "obs_prev": get_obs_by_code(
                data=self.data,
                code=code_prev,
                ft_window_size=self.ft_window_size,
                use_action_in_delta=self.use_action_in_delta,
                use_object_in_proprio=self.use_object_in_proprio,
                unsqueeze=False,
            ),
        }


if __name__ == "__main__":

    config_path = Path("drpo/configs/template.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = Path("data/data.hdf5")
    codes_path = Path("data/train_codes.pkl")

    dataset = DRPODataset(config=config, data_path=data_path, codes_path=codes_path)

    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4)
    for batch_idx, batch_samples in enumerate(dataloader):
        print(f"batch id: {batch_idx}")
        # print("samples curr: \n", batch_samples["obs_curr"]["code"])
        # print("samples prev: \n", batch_samples["obs_prev"]["code"])
        print("samples FT: ", batch_samples["obs_curr"]["ft"].shape)
        print("samples action: ", batch_samples["obs_curr"]["action"].shape)
        print("samples FT: ", batch_samples["obs_curr"]["proprio"].shape)
        print("samples FT: ", batch_samples["obs_curr"]["image"].shape)
        import pdb

        pdb.set_trace()
