import argparse
import yaml
import pickle
import csv

import numpy as np

from process_rollouts import process_rollouts
from utils import *

from dense_reward_partial_obs import DenseRewardPartialObs


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
        "-n",
        "--expert-rollouts-name",
        type=str,
        required=True,
        help="name of expert rollouts, e.g., expert_fd",
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


def eval_rollout(model: DenseRewardPartialObs, rollout: list) -> tuple:
    dense_rewards, dist_rewards = [], []

    for step, sample in enumerate(rollout):
        raw_obs = sample.obs
        dense_reward = model.predict_reward(raw_obs)
        dist_reward = sample.dist_reward
        dense_rewards.append(dense_reward)
        dist_rewards.append(dist_reward)

        if step % 50 == 0:
            print(
                f"step: {step:5d},\trolllout length: {len(rollout)}\tdistance reward: {dist_reward:5f},\tdense reward: {dense_reward:5f}"
            )

    return dense_rewards, dist_rewards


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = pathlib.Path(__file__).resolve().parent / "data"

    storage_dir = config["data_dir"] / config["env_name"] / config["offset"] / "rd2"
    expert_rollouts_name = pathlib.Path(args.expert_rollouts_name).stem

    with open(
        storage_dir
        / f"processed_{expert_rollouts_name}_{config['ft_window_size']}_best.pkl",
        "rb",
    ) as f:
        expert_rollout = pickle.load(f)

    # load successful rollout indices
    with open(storage_dir / f"{expert_rollouts_name}.csv") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            selected_rollout_indices = row
    succ_rollout_indices = [int(x) for x in selected_rollout_indices]

    # prepare rollouts for eval
    rollouts = process_rollouts(
        config,
        raw_expert_rollouts_path=storage_dir / f"{expert_rollouts_name}.pkl",
        sort_by_length=False,
    )

    # load model
    drpo = DenseRewardPartialObs(
        config=config, model_params_path=args.model_params_path
    )
    drpo.set_expert_demo(expert_rollout=expert_rollout)

    save_dir = pathlib.Path("media") / drpo.model_id / "vis_reward"
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, rollout in enumerate(rollouts):
        dense_rewards, dist_rewards = eval_rollout(drpo, rollout)
        x = np.arange(len(rollout))
        ys = {}
        ys["dense reward"] = dense_rewards
        if i not in succ_rollout_indices:
            name = "fail"
            ys["distance reward"] = dist_rewards
        else:
            name = "succ"

        plot_curves(
            x=x,
            ys=ys,
            save_dir=save_dir,
            title=f"rollout #{i} {name}",
            save_name=f"rollout_{i}_{name}",
        )
