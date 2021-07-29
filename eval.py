import pathlib
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import PartialObsAutoEncoder
from dataloader.partial_obs_dataset import PatialObsDataset
from utils import *
from process_rollouts import process_rollouts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="path of configuration file",
    )
    parser.add_argument(
        "--rollouts",
        type=str,
        default="data/lap-joint/expert_raw.pkl",
        help="raw rollouts .pkl file from RD2 project",
    )
    parser.add_argument(
        "--model-params-path", type=str, help="path of model parameter .pt file",
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


def main(config: dict, rollout: list, model_params_path: pathlib.Path) -> tuple:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PartialObsAutoEncoder(config).double().to(device)
    ckpt = torch.load(model_params_path)
    model.load_state_dict(ckpt["model_state_dict"])

    # start_obs = rollout[config["ft_window_size"] - 1].obs
    start_obs = rollout[0].obs
    goal_obs = rollout[-1].obs

    if device.type == "cuda":
        start_obs, goal_obs = to_cuda(start_obs), to_cuda(goal_obs)

    dense_rewards = []
    dist_rewards = []
    for i, step in enumerate(rollout):
        obs = step.obs
        if device.type == "cuda":
            obs = to_cuda(obs)
        dense_reward, _ = predict_reward(
            model, obs=obs, start_obs=start_obs, goal_obs=goal_obs
        )
        dist_reward = step.dist_reward

        dense_rewards.append(dense_reward)
        dist_rewards.append(dist_reward)

    return dense_rewards, dist_rewards


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    rollouts = process_rollouts(
        config=config,
        raw_expert_rollouts_pkl=pathlib.Path(args.rollouts),
        raw_expert_rollouts_csv=None,
        save=False,
        sort_by_length=False,
    )

    rollout = rollouts[2]
    rollout_len = len(rollout)
    dense_rewards, dist_rewards = main(
        config=config, rollout=rollout, model_params_path=args.model_params_path
    )

    for idx in range(rollout_len):
        if idx % 50 == 0 or idx == rollout_len - 1:
            print(
                f"distance reward: {dist_rewards[idx]:.05f}, \tdense reward: {dense_rewards[idx]:.05f}"
            )
