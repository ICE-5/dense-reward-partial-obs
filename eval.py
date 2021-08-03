import argparse
import yaml
import pickle

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
        "--raw-expert-rollouts",
        type=str,
        default="data/lap-joint/expert_raw.pkl",
        help="path of raw rollouts .pkl file generated from RD2 project",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        required=True,
        help="path of model parameter .pt file",
    )

    args = parser.parse_args()
    return args


def eval_rollout(args, expert_rollout, rollout) -> tuple:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    drpo = DenseRewardPartialObs(
        config=config, model_id=args.model_id, model_params=args.model_params
    )
    drpo.set_expert_demo(expert_rollout=expert_rollout)

    dense_rewards, dist_rewards = [], []

    for step, sample in enumerate(rollout):
        raw_obs = sample.obs
        dense_reward = drpo.predict_reward(raw_obs)
        dist_reward = sample.dist_reward
        dense_rewards.append(dense_reward)
        dist_rewards.append(dist_reward)

        if step % 50 == 0:
            print(
                f"step: {step:5d},\tdistance reward: {dist_reward:5f},\tdense reward: {dense_reward:5f}"
            )

    return dense_rewards, dist_rewards


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = pathlib.Path(__file__).resolve().parent / "data"

    rollouts = process_rollouts(config,)

    expert_rollout = rollouts[0]
    eval_succ_rollout = rollouts[1]
    eval_fail_rollout = rollouts[-1]

    for i, rollout in enumerate([eval_succ_rollout, eval_fail_rollout]):
        dense_rewards, dist_rewards = eval_rollout(args, expert_rollout, rollout)
        x = np.arange(len(rollout))
        ys = {}
        ys["dense reward"] = dense_rewards
        if i == 0:
            name = "succ"
            ys["distance reward"] = dist_rewards
        else:
            name = "fail"

        plot_curves(
            x=x, ys=ys, save_dir="media/", save_name=f"eval_{args.model_id}_{name}",
        )

