import csv
import h5py
import numpy as np
import pathlib
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from drpo.models.model import DRPONetwork
from drpo.dataloader.dataset import DRPODataset
from drpo.dataloader.utils import to_cuda, get_demo_endpoint_code, get_obs_by_code

# from drpo.utils import


class DRPO:
    """
    A wrapper class for external learning and inference
    """

    def __init__(
        self,
        config: dict,
        model_id: (str or None) = None,
        model_params_path: (str or pathlib.Path) = None,
    ) -> None:
        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and config["use_gpu"]) else "cpu"
        )
        self.config = config

        # model
        self.model_id = model_id
        self.model = DRPONetwork(config).double().to(self.device)

        # dataset
        self.data_dir = pathlib.Path(config["data_dir"])
        train_dataset = DRPODataset(
            config=config,
            data_path=self.data_dir / "data.hdf5",
            codes_path=self.data_dir / "train_codes.pkl",
        )
        test_dataset = DRPODataset(
            config=config,
            data_path=self.data_dir / "data.hdf5",
            codes_path=self.data_dir / "test_codes.pkl",
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )

        # architecture sanity check
        self.architecture = config["architecture"]
        assert self.architecture in [1, 2, 3]

        # sensor / modality
        self.sensors = config["sensors"]
        self.ft_window_size = config["ft_window_size"]

        # BEST: change to better names
        self.loss_keys = None

        # try and parse model ID from model params path
        if not model_id:
            try:
                tmp_model_id = pathlib.Path(model_params_path).parent.stem
                if "-" in tmp_model_id:
                    self.model_id = tmp_model_id
            except Exception:
                print(">>>>> Provide model_id to identify logging and plotting")
                self.model_id = "test"

        # load model params if provided
        if model_params_path:
            ckpt = torch.load(model_params_path)
            self.model.load_state_dict(ckpt["model_state_dict"])

        # TODO: remove
        self.prev_delta_z_sum = 0.0

    def train(
        self,
    ) -> None:
        # init logging service
        self.model_log_path = pathlib.Path("logs") / self.model_id
        self.model_log_path.mkdir(parents=True, exist_ok=True)
        self.model_save_dir = pathlib.Path("checkpoints") / self.model_id
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # set up logging
        tb_writer = SummaryWriter(self.model_log_path)

        with open(self.model_log_path / f"{self.model_id}.csv", "w") as f:
            writer = csv.writer(f)

        # TODO: change

        # optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        print("Training started")
        iters, epoch = 0, 0
        while iters < self.config["num_iters"]:
            epoch += 1
            for batch_sample in self.train_dataloader:
                self.model.train(True)
                iters += 1
                obs_prev, obs_curr = (
                    batch_sample["obs_prev"],
                    batch_sample["obs_curr"],
                )

                if self.device.type == "cuda":
                    obs_prev, obs_curr = (
                        to_cuda(obs_prev),
                        to_cuda(obs_curr),
                    )

                optimizer.zero_grad()
                encoded_prev = self.model(obs_prev)
                encoded_curr = self.model(obs_curr)
                z_prev, delta_z_prev = encoded_prev
                z_curr, delta_z_curr = encoded_curr

                decoded_prev = {}
                decoded_curr = {}

                if self.architecture == 1:
                    # decode ft and other sensors
                    for sensor in self.sensors:
                        if sensor == "ft":
                            decoded_prev[sensor] = self.model.decode(
                                delta_z_prev, sensor
                            )
                            decoded_curr[sensor] = self.model.decode(
                                delta_z_curr, sensor
                            )
                        else:
                            decoded_prev[sensor] = self.model.decode(z_prev, sensor)
                            decoded_curr[sensor] = self.model.decode(z_curr, sensor)

                else:
                    # only decode other sensors, not FT
                    for sensor in self.sensors:
                        if sensor != "ft":
                            decoded_prev[sensor] = self.model.decode(z_prev, sensor)

                    crafted_z_curr = z_prev + delta_z_curr
                    for sensor in self.sensors:
                        if sensor != "ft":
                            decoded_curr[sensor] = self.model.decode(
                                crafted_z_curr, sensor
                            )

                train_loss_dict = self.model.compute_loss(
                    obs_prev=obs_prev,
                    encoded_prev=encoded_prev,
                    decoded_prev=decoded_prev,
                    obs_curr=obs_curr,
                    encoded_curr=encoded_curr,
                    decoded_curr=decoded_curr,
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

                    for batch_sample in self.test_dataloader:
                        obs_prev, obs_curr = (
                            batch_sample["obs_prev"],
                            batch_sample["obs_curr"],
                        )

                        if self.device.type == "cuda":
                            obs_prev, obs_curr = (
                                to_cuda(obs_prev),
                                to_cuda(obs_curr),
                            )

                        with torch.no_grad():
                            encoded_prev = self.model(obs_prev)
                            encoded_curr = self.model(obs_curr)
                            z_prev, delta_z_prev = encoded_prev
                            z_curr, delta_z_curr = encoded_curr

                            decoded_prev = {}
                            decoded_curr = {}

                            if self.architecture == 1:
                                for sensor in self.sensors:
                                    if sensor == "ft":
                                        decoded_prev[sensor] = self.model.decode(
                                            delta_z_prev, sensor
                                        )
                                        decoded_curr[sensor] = self.model.decode(
                                            delta_z_curr, sensor
                                        )
                                    else:
                                        decoded_prev[sensor] = self.model.decode(
                                            z_prev, sensor
                                        )
                                        decoded_curr[sensor] = self.model.decode(
                                            z_curr, sensor
                                        )

                            else:
                                for sensor in self.sensors:
                                    if sensor != "ft":
                                        decoded_prev[sensor] = self.model.decode(
                                            z_prev, sensor
                                        )

                                crafted_z_curr = z_prev + delta_z_curr
                                for sensor in self.sensors:
                                    if sensor != "ft":
                                        decoded_curr[sensor] = self.model.decode(
                                            crafted_z_curr, sensor
                                        )

                            test_loss_dict = self.model.compute_loss(
                                obs_prev=obs_prev,
                                encoded_prev=encoded_prev,
                                decoded_prev=decoded_prev,
                                obs_curr=obs_curr,
                                encoded_curr=encoded_curr,
                                decoded_curr=decoded_curr,
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

    def set_init_goal_reference(self, demo_name: str) -> None:
        # load code and data
        with open(self.data_dir / "codes.pkl", "rb") as p:
            codes = pickle.load(p)
        data = h5py.File(self.data_dir / "data.hdf5", "r")

        init_code = get_demo_endpoint_code(
            codes, demo_name=demo_name, endpoint_type="init"
        )
        goal_code = get_demo_endpoint_code(
            codes, demo_name=demo_name, endpoint_type="goal"
        )

        obs_init = get_obs_by_code(
            data=data,
            code=init_code,
            ft_window_size=self.ft_window_size,
            unsqueeze=True,
        )
        obs_goal = get_obs_by_code(
            data=data,
            code=goal_code,
            ft_window_size=self.ft_window_size,
            unsqueeze=True,
        )

        data.close()

        if self.device.type == "cuda":
            obs_init = to_cuda(obs_init)
            obs_goal = to_cuda(obs_goal)

        self.model.eval()
        with torch.no_grad():
            encoded_init = self.model(obs_init)
            encoded_goal = self.model(obs_goal)
            self.z_init, _ = encoded_init
            self.z_goal, _ = encoded_goal

            # squeeze for consistency
            self.z_init = torch.squeeze(self.z_init)
            self.z_goal = torch.squeeze(self.z_goal)

    def predict_reward(self, obs: dict) -> float:
        # obs = process_raw_sample_obs(self.config, raw_obs, unsqueeze=True)
        if self.device.type == "cuda":
            obs = to_cuda(obs)

        self.model.eval()
        with torch.no_grad():
            encoded = self.model(obs)
            z, delta_z = encoded

            # squeeze for consistency
            z = torch.squeeze(z)
            delta_z = torch.squeeze(delta_z)

        # if not use_delta:
        dist_pred_g = self.calc_dist(self.z_goal, z)

        # else:
        #     reconstructed_z = self.z_init + self.prev_delta_z_sum + delta_z
        #     dist_pred_g = self.calc_dist(self.z_goal, reconstructed_z)

        dist_i_g = self.calc_dist(self.z_goal, self.z_init)
        reward = 1.0 - dist_pred_g / dist_i_g

        self.prev_delta_z_sum += delta_z
        return reward.item()

    @staticmethod
    def calc_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.dot(x, y) / (
            torch.norm(x, keepdim=True) * torch.norm(y, keepdim=True)
        )
