import pathlib
import argparse
import yaml

import torch
import torch.nn as nn

from models.model import PartialObsAutoEncoder
from process_rollouts import process_rollouts
from dataloader.utils import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path of configuration file",
    )
    parser.add_argument(
        "--raw-expert-rollouts-pkl",
        type=str,
        default="data/lap-joint/expert_raw.pkl",
        help="path of raw rollouts .pkl file generated from RD2 project",
    )
    parser.add_argument(
        "--raw-expert-rollouts-csv",
        type=str,
        default="data/lap-joint/expert_raw.csv",
        help="path of .csv file specifying indices of successful rollouts in .pkl file",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        required=True,
        help="path of model parameter .pt file",
    )

    args = parser.parse_args()
    return args


def predict_reward(
    model: nn.Module, obs: dict, start_obs: dict, goal_obs: dict
) -> float:
    model.eval()

    with torch.no_grad():
        z_obs, _ = model(obs)
        z_start, _ = model(start_obs)
        z_goal, _ = model(goal_obs)

        z_obs = torch.squeeze(z_obs)
        z_start = torch.squeeze(z_start)
        z_goal = torch.squeeze(z_goal)

        dist_s_g = 1.0 - torch.dot(z_goal, z_start)
        dist_pred_g = 1.0 - torch.dot(z_goal, z_obs)
        reward = 1.0 - dist_pred_g / dist_s_g

    return reward, (z_start, z_goal, start_obs, goal_obs)


def main(
    config: dict, rollout: list, successful_rollout: list, model_params: pathlib.Path
) -> tuple:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PartialObsAutoEncoder(config).double().to(device)
    ckpt = torch.load(model_params)
    model.load_state_dict(ckpt["model_state_dict"])

    # start_obs = rollout[config["ft_window_size"] - 1].obs
    start_raw_obs = successful_rollout[0].obs
    goal_raw_obs = successful_rollout[-1].obs

    # process obs
    start_obs = process_raw_sample_obs(config, start_raw_obs, unsqueeze=True)
    goal_obs = process_raw_sample_obs(config, goal_raw_obs, unsqueeze=True)

    if device.type == "cuda":
        start_obs, goal_obs = to_cuda(start_obs), to_cuda(goal_obs)

    dense_rewards = []
    dist_rewards = []
    for step, sample in enumerate(rollout):
        raw_obs = sample.obs
        obs = process_raw_sample_obs(config, raw_obs, unsqueeze=True)
        if device.type == "cuda":
            obs = to_cuda(obs)

        dense_reward, _ = predict_reward(
            model, obs=obs, start_obs=start_obs, goal_obs=goal_obs
        )
        dist_reward = sample.dist_reward

        dense_rewards.append(dense_reward)
        dist_rewards.append(dist_reward)

        if step % 50 == 0 or step == len(rollout) - 1:
            print(
                f"step:{step:05d},\tdistance reward: {dist_reward:.05f},\tdense reward: {dense_reward:.05f}"
            )
    print("\n")

    return dense_rewards, dist_rewards


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    rollouts = process_rollouts(
        config=config,
        raw_expert_rollouts_pkl=pathlib.Path(args.raw_expert_rollouts_pkl),
        raw_expert_rollouts_csv=None,
        save=False,
        sort_by_length=True,
    )

    successful_rollouts = process_rollouts(
        config=config,
        raw_expert_rollouts_pkl=pathlib.Path(args.raw_expert_rollouts_pkl),
        raw_expert_rollouts_csv=pathlib.Path(args.raw_expert_rollouts_csv),
        save=False,
        sort_by_length=True,
    )
    
    successful_rollout = successful_rollouts[0]

    # test successful rollout
    a_rollout = successful_rollouts[1]
    a_dense_rewards, a_dist_rewards = main(
        config=config, rollout=a_rollout, successful_rollout=successful_rollout, model_params=args.model_params
    )

    # test failed rollout
    b_rollout = rollouts[-1]
    b_dense_rewards, b_dist_rewards = main(
        config=config, rollout=b_rollout, successful_rollout=successful_rollout, model_params=args.model_params
    )

    experiment_id = args.model_params.split("/")[-2]
    plot_curves(
        x=np.arange(len(a_rollout)),
        ys={"dense rewared": a_dense_rewards,},
        save_dir="media/",
        save_name=f"eval_succ_{experiment_id}",
    )

    plot_curves(
        x=np.arange(len(b_rollout)),
        ys={"dense rewared": b_dense_rewards, "dist reward": b_dist_rewards},
        save_dir="media/",
        save_name=f"eval_fail_{experiment_id}",
    )

