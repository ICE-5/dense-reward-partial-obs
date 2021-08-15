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
        self,
        config: dict,
        model_id: str or None = None,
        model_params_path: str or pathlib.Path = None,
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
                tmp_model_id = pathlib.Path(model_params_path).parent.stem
                if "-" in tmp_model_id:
                    self.model_id = tmp_model_id
            except Exception:
                print(
                    ">>>>> please try to provide a model id for distinguishing logging and plotting"
                )
                self.model_id = "test"

        if model_params_path is not None:
            ckpt = torch.load(model_params_path)
            self.model.load_state_dict(ckpt["model_state_dict"])

    def train(self,) -> None:
        # init logging service
        self.model_log_path = pathlib.Path("logs") / self.model_id
        self.model_log_path.mkdir(parents=True, exist_ok=True)
        self.model_save_dir = pathlib.Path("checkpoints") / self.model_id
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # set up logging
        tb_writer = SummaryWriter(self.model_log_path)

        with open(self.model_log_path / f"{self.model_id}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "train_loss",
                    "train_recon_loss",
                    "train_tmp_loss",
                    "test_loss",
                    "test_recon_loss",
                    "test_tmp_loss",
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
                obs_curr, obs_next = (
                    batch_sample["obs_curr"],
                    batch_sample["obs_next"],
                )

                if self.device.type == "cuda":
                    obs_curr, obs_next = (
                        to_cuda(obs_curr),
                        to_cuda(obs_next),
                    )

                optimizer.zero_grad()
                encoded_curr, decoded_curr = self.model(obs_curr)
                encoded_next, decoded_next = self.model(obs_next)

                train_loss, train_recon_loss, train_tmp_loss = self.model.compute_loss(
                    obs_curr=obs_curr,
                    encoded_curr=encoded_curr,
                    decoded_curr=decoded_curr,
                    obs_next=obs_next,
                    encoded_next=encoded_next,
                    decoded_next=decoded_next,
                )

                # write to tensorboard
                tb_writer.add_scalar("Loss/loss_all", train_loss, iters)
                tb_writer.add_scalar("Loss/loss_reconstruct", train_recon_loss, iters)
                tb_writer.add_scalar(
                    "Loss/loss_temporal_enforce", train_tmp_loss, iters
                )

                # log in terminal output
                if iters % self.config["log_freq"] == 0:

                    test_losses, test_recon_losses, test_tmp_losses = [], [], []
                    for batch_sample in test_dataloader:
                        obs_curr, obs_next = (
                            batch_sample["obs_curr"],
                            batch_sample["obs_next"],
                        )

                        if self.device.type == "cuda":
                            obs_curr, obs_next = (
                                to_cuda(obs_curr),
                                to_cuda(obs_next),
                            )

                        with torch.no_grad():
                            encoded_curr, decoded_curr = self.model(obs_curr)
                            encoded_next, decoded_next = self.model(obs_next)

                            loss, recon_loss, tmp_loss = self.model.compute_loss(
                                obs_curr=obs_curr,
                                encoded_curr=encoded_curr,
                                decoded_curr=decoded_curr,
                                obs_next=obs_next,
                                encoded_next=encoded_next,
                                decoded_next=decoded_next,
                            )

                            test_losses.append(loss.item())
                            test_recon_losses.append(recon_loss.item())
                            test_tmp_losses.append(tmp_loss.item())

                    test_loss = np.mean(test_losses)
                    test_recon_loss = np.mean(test_recon_losses)
                    test_tmp_loss = np.mean(test_tmp_losses)

                    with open(self.model_log_path / f"{self.model_id}.csv", "a") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                train_loss.item(),
                                train_recon_loss.item(),
                                train_tmp_loss.item(),
                                test_loss,
                                test_recon_loss,
                                test_tmp_loss,
                            ]
                        )

                    tb_writer.add_scalar("Loss/test_loss", test_loss, iters)
                    tb_writer.add_scalar(
                        "Loss/test_loss_reconstruct", test_recon_loss, iters
                    )
                    tb_writer.add_scalar(
                        "Loss/test_loss_temporal_enforce", test_tmp_loss, iters
                    )

                    print(
                        f"Iter: {iters:7d}\tTrain Loss: {train_loss.item():0.3f}\tTrain Recon Loss: {train_recon_loss.item():0.3f}\tTrain Temp Enforce Loss: {train_tmp_loss.item():0.3f}"
                        + f"\tTest Loss: {test_loss.item():0.3f}\tTest Recon Loss: {test_recon_loss.item():0.3f}\tTest Temp Enforce Loss: {test_tmp_loss.item():0.3f}"
                    )

                # perform update
                train_loss.backward()
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
                        self.model_save_dir / f"{iters}.pt",
                    )

                if iters > self.config["num_iters"]:
                    break

    def set_expert_demo(self, expert_rollout: list,) -> None:
        raw_obs_init = expert_rollout[0].obs
        raw_obs_goal = expert_rollout[-1].obs

        # process obs
        obs_init = process_raw_sample_obs(self.config, raw_obs_init, unsqueeze=True)
        obs_goal = process_raw_sample_obs(self.config, raw_obs_goal, unsqueeze=True)

        if self.device.type == "cuda":
            obs_init = to_cuda(obs_init)
            obs_goal = to_cuda(obs_goal)

        self.model.eval()
        with torch.no_grad():
            encoded_init, _ = self.model(obs_init)
            encoded_goal, _ = self.model(obs_goal)
            self.z_init, _ = encoded_init
            self.z_goal, _ = encoded_goal
            self.z_init = torch.squeeze(self.z_init)
            self.z_goal = torch.squeeze(self.z_goal)

    def predict_reward(self, raw_obs: dict) -> float:
        obs = process_raw_sample_obs(self.config, raw_obs, unsqueeze=True)
        if self.device.type == "cuda":
            obs = to_cuda(obs)

        self.model.eval()
        with torch.no_grad():
            encoded, _ = self.model(obs)
            z, _ = encoded
            z = torch.squeeze(z)

            dist_s_g = 1.0 - torch.dot(self.z_goal, self.z_init)
            dist_pred_g = 1.0 - torch.dot(self.z_goal, z)
            reward = 1.0 - dist_pred_g / dist_s_g

        return reward.item()

