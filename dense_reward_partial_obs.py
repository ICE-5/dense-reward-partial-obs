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

        self.architecture = config["architecture"]
        self.sensors = config["sensor_used_in_model"]
        self.loss_keys = None

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

        self.prev_delta_z_sum = 0.0

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
                encoded_curr = self.model(obs_curr)
                encoded_next = self.model(obs_next)
                z_curr, delta_z_curr = encoded_curr
                z_next, delta_z_next = encoded_next

                decoded_curr = {}
                decoded_next = {}

                if self.architecture == 1:
                    for sensor in self.sensors:
                        if sensor == "ft":
                            decoded_curr[sensor] = self.model.decode(
                                delta_z_curr, sensor
                            )
                            decoded_next[sensor] = self.model.decode(
                                delta_z_next, sensor
                            )
                        else:
                            decoded_curr[sensor] = self.model.decode(z_curr, sensor)
                            decoded_next[sensor] = self.model.decode(z_next, sensor)

                elif self.architecture == 2 or self.architecture == 3:
                    for sensor in self.sensors:
                        if sensor != "ft":
                            decoded_curr[sensor] = self.model.decode(z_curr, sensor)

                    crafted_z_next = z_curr + delta_z_next
                    for sensor in self.sensors:
                        if sensor != "ft":
                            decoded_next[sensor] = self.model.decode(
                                crafted_z_next, sensor
                            )

                else:
                    raise ValueError("Invalid architecture type")

                train_loss_dict = self.model.compute_loss(
                    obs_curr=obs_curr,
                    encoded_curr=encoded_curr,
                    decoded_curr=decoded_curr,
                    obs_next=obs_next,
                    encoded_next=encoded_next,
                    decoded_next=decoded_next,
                )

                if iters == 1:
                    self.loss_keys = train_loss_dict.keys()

                # write to tensorboard
                for k in self.loss_keys:
                    tb_writer.add_scalar(f"Loss_train/{k}", train_loss_dict[k], iters)

                # log in terminal output
                if iters % self.config["log_freq"] == 0:

                    output = []
                    for k in self.loss_keys:
                        output.append(train_loss_dict[k].item())

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
                            encoded_curr = self.model(obs_curr)
                            encoded_next = self.model(obs_next)
                            z_curr, delta_z_curr = encoded_curr
                            z_next, delta_z_next = encoded_next

                            decoded_curr = {}
                            decoded_next = {}

                            if self.architecture == 1:
                                for sensor in self.sensors:
                                    if sensor == "ft":
                                        decoded_curr[sensor] = self.model.decode(
                                            delta_z_curr, sensor
                                        )
                                        decoded_next[sensor] = self.model.decode(
                                            delta_z_next, sensor
                                        )
                                    else:
                                        decoded_curr[sensor] = self.model.decode(z_curr, sensor)
                                        decoded_next[sensor] = self.model.decode(z_next, sensor)

                            elif self.architecture == 2 or self.architecture == 3:
                                for sensor in self.sensors:
                                    if sensor != "ft":
                                        decoded_curr[sensor] = self.model.decode(z_curr, sensor)

                                crafted_z_next = z_curr + delta_z_next
                                for sensor in self.sensors:
                                    if sensor != "ft":
                                        decoded_next[sensor] = self.model.decode(
                                            crafted_z_next, sensor
                                        )

                            else:
                                raise ValueError("Invalid architecture type")

                            test_loss_dict = self.model.compute_loss(
                                obs_curr=obs_curr,
                                encoded_curr=encoded_curr,
                                decoded_curr=decoded_curr,
                                obs_next=obs_next,
                                encoded_next=encoded_next,
                                decoded_next=decoded_next,
                            )

                            for k in self.loss_keys:
                                if f"test_{k}_list" not in globals().keys():
                                    globals()[f"test_{k}_list"] = []

                                eval(f"test_{k}_list").append(test_loss_dict[k].item())

                    for k in self.loss_keys:
                        globals()[f"test_{k}"] = np.mean(eval(f"test_{k}_list"))
                        output.append(eval(f"test_{k}"))
                        del globals()[f"test_{k}_list"]

                    with open(self.model_log_path / f"{self.model_id}.csv", "a") as f:
                        writer = csv.writer(f)
                        writer.writerow(output)

                    for k in self.loss_keys:
                        tb_writer.add_scalar(f"Loss_test/{k}", eval(f"test_{k}"), iters)

                    # print to terminal
                    refactored_output = f"iter: {iters:6d}\t"
                    for i, k in enumerate(self.loss_keys):
                        refactored_output += f"train_{k}: {output[i]:.4f}\t"
                    for i, k in enumerate(self.loss_keys):
                        refactored_output += f"test_{k}: {output[i+3]:.4f}\t"
                    print(refactored_output)

                # perform update
                train_loss_dict["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10, "inf")
                optimizer.step()

                # save model
                if iters % self.config["save_freq"] == 0:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": train_loss_dict["loss"],
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
            encoded_init = self.model(obs_init)
            encoded_goal = self.model(obs_goal)
            self.z_init, _ = encoded_init
            self.z_goal, _ = encoded_goal
            self.z_init = torch.squeeze(self.z_init)
            self.z_goal = torch.squeeze(self.z_goal)

    def predict_reward(self, raw_obs: dict, use_delta: bool = False) -> float:
        obs = process_raw_sample_obs(self.config, raw_obs, unsqueeze=True)
        if self.device.type == "cuda":
            obs = to_cuda(obs)

        self.model.eval()
        with torch.no_grad():
            encoded = self.model(obs)
            z, delta_z = encoded
            z = torch.squeeze(z)
            delta_z = torch.squeeze(delta_z)

        if not use_delta:
            dist_pred_g = 1.0 - torch.dot(self.z_goal, z)

        else:
            reconstructed_z = self.z_init + self.prev_delta_z_sum + delta_z
            reconstructed_z = reconstructed_z / torch.norm(
                reconstructed_z, keepdim=True
            )
            dist_pred_g = 1.0 - torch.dot(self.z_goal, reconstructed_z)

        dist_s_g = 1.0 - torch.dot(self.z_goal, self.z_init)
        reward = 1.0 - dist_pred_g / dist_s_g

        self.prev_delta_z_sum += delta_z
        return reward.item()

