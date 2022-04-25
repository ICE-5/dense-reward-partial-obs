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
    # parser.add_argument(
    #     "-n",
    #     "--expert-rollouts-name",
    #     type=str,
    #     required=True,
    #     help="name of expert rollouts, e.g., expert_fd",
    # )
    parser.add_argument(
        "-m",
        "--model-params-path",
        type=str,
        required=True,
        help="path of model parameter .pt file",
    )
    # parser.add_argument(
    #     "--use-delta",
    #     type=bool,
    #     default=False,
    #     required=False,
    #     help="whether to use delta hidden state to infer reward",
    # )

    args = parser.parse_args()
    return args


def eval_reward(model: DRPO, data_path: Path, codes_path: Path, demo_name: str) -> list:
    data = h5py.File(data_path, "r")
    with open(codes_path, "rb") as p:
        codes = pickle.load(p)
    demo_codes = get_demo_codes_by_name(codes, demo_name)

    dense_reward_arr = []
    original_reward_arr = data[f"data/{demo_name}/0/reward"][()]
    for code in demo_codes:
        obs = get_obs_by_code(
            data=data,
            code=code,
            ft_window_size=model.ft_window_size,
            use_action_in_delta=True,
            unsqueeze=True,
        )
        dense_reward = model.predict_reward(obs)
        dense_reward_arr.append(dense_reward)
    return dense_reward_arr, original_reward_arr


# def eval_rollout(
#     model: DenseRewardPartialObs, rollout: list, use_delta: bool = False
# ) -> tuple:
#     dense_rewards, dist_rewards = [], []

#     for step, sample in enumerate(rollout):
#         raw_obs = sample.obs

#         if use_delta and step == 0:
#             model.prev_delta_z_sum = 0.0

#         dense_reward = model.predict_reward(raw_obs, use_delta)
#         dist_reward = sample.dist_reward
#         dense_rewards.append(dense_reward)
#         dist_rewards.append(dist_reward)

#         # if step % 100 == 0 or step == len(rollout) - 1:
#         if step == len(rollout) - 1:
#             print(
#                 f"step: {step:5d},\trolllout length: {len(rollout)}\tdistance reward: {dist_reward:5f},\tdense reward: {dense_reward:5f}"
#             )

#     return dense_rewards, dist_rewards


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # storage_dir = config["data_dir"] / config["env_name"] / config["offset"] / "rd2"
    # expert_rollouts_name = pathlib.Path(args.expert_rollouts_name).stem

    # with open(
    #     storage_dir
    #     / f"processed_{expert_rollouts_name}_{config['ft_window_size']}_best.pkl",
    #     "rb",
    # ) as f:
    #     expert_rollout = pickle.load(f)

    # # load successful rollout indices
    # with open(storage_dir / f"{expert_rollouts_name}.csv") as f:
    #     reader = csv.reader(f, delimiter=",")
    #     for row in reader:
    #         selected_rollout_indices = row
    # succ_rollout_indices = [int(x) for x in selected_rollout_indices]

    # # prepare rollouts for eval
    # rollouts = process_rollouts(
    #     config,
    #     raw_expert_rollouts_path=storage_dir / f"{expert_rollouts_name}.pkl",
    #     sort_by_length=False,
    # )

    demo_name = "demo_1"
    data_dir = Path(config["data_dir"])

    # load model
    model = DRPO(config=config, model_id=None, model_params_path=args.model_params_path)
    model.set_init_goal_reference(demo_name=demo_name)

    print(model.model_id)

    dense_reward_arr, original_reward_arr = eval_reward(
        model=model,
        data_path=data_dir / "data.hdf5",
        codes_path=data_dir / "codes.pkl",
        demo_name=demo_name,
    )

    save_dir = pathlib.Path("media") / model.model_id / "vis_reward_new"
    save_dir.mkdir(parents=True, exist_ok=True)

    # for i, rollout in enumerate(rollouts):
    #     dense_rewards, dist_rewards = eval_rollout(drpo, rollout, args.use_delta)
    #     x = np.arange(len(rollout))
    #     ys = {}
    #     ys["dense reward"] = dense_rewards
    #     if i not in succ_rollout_indices:
    #         name = "fail"
    #         # ys["distance reward"] = dist_rewards
    #     else:
    #         name = "succ"
    x = np.arange(len(dense_reward_arr))
    ys = {}
    ys["dense reward"] = dense_reward_arr
    ys["original reward"] = original_reward_arr

    plot_curves(
        x=x,
        ys=ys,
        save_dir=save_dir,
        title=f"{demo_name}",
        save_name=f"{demo_name}",
    )

    prGreen(f"\nSUCCESS | directory with plots: {save_dir}\n")
