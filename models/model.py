import torch
import torch.nn as nn
import numpy as np

from models.base_models.encoders import DepthmapEncoder, ImageEncoder, FusionNet
from models.base_models.decoders import DepthmapDecoder, ImageDecoder


class PartialObsAutoEncoder(nn.Module):
    def __init__(self, config: dict) -> None:

        super().__init__()

        self.ft_network_type = config["ft_network_type"]

        # security check before running exec
        if self.ft_network_type not in ["MLP", "LSTM"]:
            raise ValueError("Invalid network type, dangerous input")
        exec(
            f"from models.base_models.encoders import FtEncoder{self.ft_network_type} as FtEncoder"
        )
        exec(
            f"from models.base_models.decoders import FtDecoder{self.ft_network_type} as FtDecoder"
        )

        self.config = config
        self.z_dim = config["z_dim"]
        initialize_weights = config["initialize_weights"]
        self.sensors = config["sensor_used_in_model"]

        # there must be FT sensor to provide delta_z info
        assert "ft" in self.sensors
        # there must be another sensor other than FT to provide z info
        assert len(self.sensors) > 1
        # there are multiple other sensors
        if len(self.sensors) > 2:
            self.num_other_sensors = len(self.sensors) - 1
            self.fusion = FusionNet(
                self.num_other_sensors * self.z_dim, self.z_dim, initialize_weights
            )

        # the sensor should always include ft, plus other sensor such as depthmap
        for sensor in self.sensors:
            if sensor == "ft":
                params = {
                    "ft_window_size": self.config["ft_window_size"],
                }
            else:
                params = {}

            # encoder
            # TODO: add adaptibility for image and depth
            setattr(
                self,
                f"{sensor}_encoder",
                eval(f"{sensor.capitalize()}Encoder")(
                    self.z_dim, initialize_weights, **params
                ),
            )

            # decoder
            setattr(
                self,
                f"{sensor}_decoder",
                eval(f"{sensor.capitalize()}Decoder")(
                    self.z_dim, initialize_weights, **params
                ),
            )

        # TODO: try cross-entropy loss
        self.l2_loss = nn.MSELoss()

    def encode(self, obs: dict) -> dict:
        raw_encoded = {}
        for sensor in self.sensors:
            x = obs[sensor]
            out = getattr(self, f"{sensor}_encoder")(x)
            raw_encoded[sensor] = out
        return raw_encoded

    def process_raw_encoded(self, raw_encoded: dict) -> tuple:
        delta_z = raw_encoded["ft"]

        if len(self.sensors) > 2:
            concat_z = []
            for sensor in self.sensors:
                if sensor != "ft":
                    concat_z.append(raw_encoded[sensor])
            z = self.fusion(torch.cat(concat_z, dim=1))
            z = z / torch.norm(z, dim=1, keepdim=True)
        else:
            for sensor in self.sensors:
                if sensor != "ft":
                    z = raw_encoded[sensor]
                    z = z / torch.norm(z, dim=1, keepdim=True)

        return z, delta_z

    def decode(self, z, delta_z) -> dict:
        decoded = {}
        for sensor in self.sensors:
            if sensor == "ft":
                decoded[sensor] = getattr(self, f"{sensor}_decoder")(delta_z)
            else:
                decoded[sensor] = getattr(self, f"{sensor}_decoder")(z)
        return decoded

    def forward(self, obs: dict) -> tuple:
        raw_encoded = self.encode(obs)
        z, delta_z = self.process_raw_encoded(raw_encoded)
        encoded = (z, delta_z)
        decoded = self.decode(z, delta_z)
        return encoded, decoded

    def compute_loss(
        self,
        obs_curr: dict,
        encoded_curr: dict,
        decoded_curr: dict,
        obs_next: dict,
        encoded_next: dict,
        decoded_next: dict,
    ) -> tuple:
        # reconstruction loss
        recon_loss_curr = self.compute_reconstruction_loss(obs_curr, decoded_curr)
        recon_loss_next = self.compute_reconstruction_loss(obs_next, decoded_next)
        recon_loss = recon_loss_curr + recon_loss_next

        # temporal enforcement loss
        temp_enforce_loss = self.compute_temporal_enforcement_loss(
            encoded_curr, encoded_next
        )

        loss = (
            self.config["reconstruction_lambda"] * recon_loss
            + self.config["temporal_enforcement_lambda"] * temp_enforce_loss
        )
        return loss, recon_loss, temp_enforce_loss

    def compute_temporal_enforcement_loss(self, encoded_curr, encoded_next) -> torch.Tensor:
        z_curr, _ = encoded_curr
        z_next, delta_z_next = encoded_next

        # utilize z_next = z_curr + delta_z
        loss = self.l2_loss(z_curr + delta_z_next, z_next)
        return loss

    def compute_reconstruction_loss(self, obs: dict, decoded: dict) -> torch.Tensor:
        loss = 0.0
        for sensor in self.config["sensor_used_in_model"]:
            loss += self.l2_loss(obs[sensor], decoded[sensor],)
        return loss

