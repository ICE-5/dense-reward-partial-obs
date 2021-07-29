import pathlib
import argparse
import yaml
from datetime import datetime
from pdb import set_trace as debug

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import PartialObsAutoEncoder
from dataloader.partial_obs_dataset import PatialObsDataset
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="path of configuration file",
    )

    args = parser.parse_args()
    return args


def train(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PartialObsAutoEncoder(config).double().to(device)
    dataset = PatialObsDataset(config, phase="train")
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    log_dir = pathlib.Path(__file__).resolve().parent / "logs" / config["experiment_id"]
    writer = SummaryWriter(log_dir)

    iters, epoch = 0, 0
    while iters < config["num_iters"]:
        epoch += 1
        # print(f"dataset length: {len(dataloader)}")
        for batch_sample in dataloader:
            iters += 1
            data_a, data_b, data_g = (
                batch_sample["a_obs"],
                batch_sample["b_obs"],
                batch_sample["goal_obs"],
            )

            if device.type == "cuda":
                data_a, data_b, data_g = (
                    to_cuda(data_a),
                    to_cuda(data_b),
                    to_cuda(data_g),
                )

            optimizer.zero_grad()
            z_a, decoded_a = model(data_a)
            z_b, decoded_b = model(data_b)
            z_g, _ = model(data_g)

            loss, recon_loss, cmp_loss = model.compute_loss(
                data_a, decoded_a, z_a, data_b, decoded_b, z_b, z_g
            )
            # write to tensorboard
            writer.add_scalar("Loss/loss_all", loss, iters)
            writer.add_scalar("Loss/loss_reconstruct", recon_loss, iters)
            writer.add_scalar("Loss/loss_comparison", cmp_loss, iters)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, "inf")
            optimizer.step()

            # log in terminal output
            if iters % config["log_freq"] == 0:
                print(
                    f"Iter: {iters:8d} Loss: {loss.item():0.3f} Reconstruct Loss: {recon_loss.item():0.3f} Comparison Loss: {cmp_loss.item():0.3f}"
                )
                

            # save model
            if iters % config["save_freq"] == 0:
                save_dir = (
                    pathlib.Path(__file__).resolve().parent
                    / "checkpoints"
                    / config["experiment_id"]
                )
                save_dir.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    save_dir / f"{iters:8d}.pt",
                )

            if iters > config["num_iters"]:
                break


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["experiment_id"] = datetime.now().strftime("%m%d%Y-%H%M%S")

    train(config)
