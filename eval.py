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
        "--config", type=str, required=True, help="path of configuration file",
    )
    parser.add_argument(
        "--model-params",
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

    rollout_dir = config["data_dir"] / config["env_name"] / config["offset"] / "rd2"

    with open(rollout_dir / "best_expert.pkl", "rb") as f:
        expert_rollout = pickle.load(f)

    # load successful rollout indices
    with open(rollout_dir / "expert_raw.csv") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            selected_rollout_indices = row
    succ_rollout_indices = [int(x) for x in selected_rollout_indices]

    # prepare rollouts for eval
    rollouts = process_rollouts(config, sort_by_length=False)

    # load model
    drpo = DenseRewardPartialObs(config=config, model_params=args.model_params)
    drpo.set_expert_demo(expert_rollout=expert_rollout)

    save_dir = pathlib.Path("media") / drpo.model_id
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
            save_name=f"eval_{drpo.model_id}_{i}_{name}",
        )

