import csv
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import *
from dataloader.partial_obs_dataset import *
from dataloader.utils import *
from utils import *


class DenseRewardPartialObs:
    """
    A wrapper class for external learning and inference
    """

    def __init__(
        self, config: dict, model_id: str or None = None, model_params: str = None
    ) -> None:
        self.model_id = model_id
        self.config = config
        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and config["use_gpu"]) else "cpu"
        )
        self.model = PartialObsAutoEncoder(config).double().to(self.device)

        # try and parse model ID from model params path
        if model_id is None:
            try:
                tmp_model_id = model_params.split("/")[-2]
                if "-" in tmp_model_id:
                    self.model_id = tmp_model_id
            except Exception:
                print(
                    ">>>>> please try to provide a model id for distinguishing logging and plotting"
                )
                self.model_id = "eval"

        if model_params is not None:
            ckpt = torch.load(model_params)
            self.model.load_state_dict(ckpt["model_state_dict"])

    def train(self,) -> None:
        # init logging service
        self.model_log_path = pathlib.Path("logs") / self.model_id
        self.model_log_path.mkdir(parents=True, exist_ok=True)
        self.model_save_path = pathlib.Path("checkpoints") / self.model_id
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # set up logging
        tb_writer = SummaryWriter(self.model_log_path)
        csv_f = open(self.model_log_path / "plot.csv", "w")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(
            [
                "train_loss",
                "train_recon_loss",
                "train_cmp_loss",
                "test_loss",
                "test_recon_loss",
                "test_cmp_loss",
            ]
        )

        # load data
        train_dataset = PatialObsDataset(self.config, phase="train")
        test_dataset = PatialObsDataset(self.config, phase="test")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

        # optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        iters, epoch = 0, 0
        while iters < self.config["num_iters"]:
            epoch += 1
            for batch_sample in train_dataloader:
                self.model.train(True)
                iters += 1
                data_a, data_b, data_g = (
                    batch_sample["a_obs"],
                    batch_sample["b_obs"],
                    batch_sample["goal_obs"],
                )

                if self.device.type == "cuda":
                    data_a, data_b, data_g = (
                        to_cuda(data_a),
                        to_cuda(data_b),
                        to_cuda(data_g),
                    )

                optimizer.zero_grad()
                z_a, decoded_a = self.model(data_a)
                z_b, decoded_b = self.model(data_b)
                z_g, _ = self.model(data_g)

                loss, recon_loss, cmp_loss = self.model.compute_loss(
                    data_a, decoded_a, z_a, data_b, decoded_b, z_b, z_g
                )

                # write to tensorboard
                tb_writer.add_scalar("Loss/loss_all", loss, iters)
                tb_writer.add_scalar("Loss/loss_reconstruct", recon_loss, iters)
                tb_writer.add_scalar("Loss/loss_comparison", cmp_loss, iters)

                # log in terminal output
                if iters % self.config["log_freq"] == 0:
                    print(
                        f"Iter: {iters:8d} Loss: {loss.item():0.3f} Reconstruct Loss: {recon_loss.item():0.3f} Comparison Loss: {cmp_loss.item():0.3f}"
                    )

                    test_losses, test_recon_losses, test_cmp_losses = [], [], []
                    for batch_sample in test_dataloader:
                        data_a, data_b, data_g = (
                            batch_sample["a_obs"],
                            batch_sample["b_obs"],
                            batch_sample["goal_obs"],
                        )

                        if self.device.type == "cuda":
                            data_a, data_b, data_g = (
                                to_cuda(data_a),
                                to_cuda(data_b),
                                to_cuda(data_g),
                            )

                        with torch.no_grad():
                            z_a, decoded_a = self.model(data_a)
                            z_b, decoded_b = self.model(data_b)
                            z_g, _ = self.model(data_g)

                            (
                                test_loss,
                                test_recon_loss,
                                test_cmp_loss,
                            ) = self.model.compute_loss(
                                data_a, decoded_a, z_a, data_b, decoded_b, z_b, z_g
                            )
                            test_losses.append(test_loss.item())
                            test_recon_losses.append(test_recon_loss.item())
                            test_cmp_losses.append(test_cmp_loss.item())

                    csv_writer.writerow(
                        [
                            loss.item(),
                            recon_loss.item(),
                            cmp_loss.item(),
                            np.mean(test_losses),
                            np.mean(test_recon_losses),
                            np.mean(test_cmp_losses),
                        ]
                    )

                # perform update
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10, "inf")
                optimizer.step()

                # save model
                if iters % self.config["save_freq"] == 0:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        self.save_dir / f"{iters}.pt",
                    )

                if iters > self.config["num_iters"]:
                    csv_writer.close()
                    break

    def set_expert_demo(self, expert_rollout: list,) -> None:
        start_raw_obs = expert_rollout[0].obs
        goal_raw_obs = expert_rollout[-1].obs

        # process obs
        start_obs = process_raw_sample_obs(self.config, start_raw_obs, unsqueeze=True)
        goal_obs = process_raw_sample_obs(self.config, goal_raw_obs, unsqueeze=True)

        if self.device.type == "cuda":
            start_obs = to_cuda(start_obs)
            goal_obs = to_cuda(goal_obs)

        self.model.eval()
        with torch.no_grad():
            self.z_start, _ = self.model(start_obs)
            self.z_goal, _ = self.model(goal_obs)
            self.z_start = torch.squeeze(self.z_start)
            self.z_goal = torch.squeeze(self.z_goal)

    def predict_reward(self, raw_obs: dict) -> float:
        obs = process_raw_sample_obs(self.config, raw_obs, unsqueeze=True)
        if self.device.type == "cuda":
            obs = to_cuda(obs)
        self.model.eval()

        with torch.no_grad():
            z_obs, _ = self.model(obs)
            z_obs = torch.squeeze(z_obs)

            dist_s_g = 1.0 - torch.dot(self.z_goal, self.z_start)
            dist_pred_g = 1.0 - torch.dot(self.z_goal, z_obs)
            reward = 1.0 - dist_pred_g / dist_s_g

        return reward.item()

