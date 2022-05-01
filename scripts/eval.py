import argparse
from matplotlib.pyplot import draw_if_interactive
import yaml
import pickle
import csv
import h5py
from pathlib import Path
from h5py._hl.files import File

import numpy as np

from drpo.drpo import DRPO
from drpo.dataloader.utils import *
from drpo.utils import *

# from process_rollouts import process_rollouts
# from utils import *

# from dense_reward_partial_obs import DenseRewardPartialObs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path of configuration file, check configs/ for template ",
    )
    parser.add_argument(
        "-m",
        "--model-params-path",
        type=str,
        required=True,
        help="path of model parameter .pt file",
    )
    args = parser.parse_args()
    return args


def eval_reward(model: DRPO, data_dir: Path, demo_name: str) -> list:
    data_dir = Path(data_dir)

    data = h5py.File(data_dir / "data.hdf5", "r")
    with open(data_dir / "codes.pkl", "rb") as p:
        codes = pickle.load(p)
    demo_codes = get_demo_codes_by_name(codes, demo_name)

    dense_reward_arr = []
    original_reward_arr = data[f"data/{demo_name}/0/reward"][()]
    for code in demo_codes:
        obs = get_obs_by_code(
            data=data,
            code=code,
            ft_window_size=model.ft_window_size,
            use_action_in_delta=model.use_action_in_delta,
            use_object_in_proprio=model.use_object_in_proprio,
            unsqueeze=True,
        )
        dense_reward = model.predict_reward(obs)
        dense_reward_arr.append(dense_reward)
    return dense_reward_arr, original_reward_arr


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    demo_name = "demo_1"
    data_dir = Path(config["data_dir"])

    # load model
    model = DRPO(config=config, model_id=None, model_params_path=args.model_params_path)
    model.set_init_goal_reference(demo_name=demo_name)

    print(model.model_id)

    dense_reward_arr, original_reward_arr = eval_reward(
        model=model,
        data_path=data_dir,
        demo_name=demo_name,
    )

    save_dir = pathlib.Path("media") / model.model_id / "vis_reward_new"
    save_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(dense_reward_arr))
    ys = {}
    ys["dense reward"] = dense_reward_arr
    ys["original reward"] = original_reward_arr

    plot_smooth_curves(
        x=x,
        ys=ys,
        save_dir=save_dir,
        save_name=f"{demo_name}",
        xlabel="Steps",
        ylabel="Rewards"
    )

    prGreen(f"\nSUCCESS | directory with plots: {save_dir}\n")
