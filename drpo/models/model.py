import torch
import torch.nn as nn
import numpy as np

from drpo.models.base_models.encoders import *
from drpo.models.base_models.decoders import *


class DRPONetwork(nn.Module):
    def __init__(self, config: dict) -> None:

        super().__init__()

        # sanity check
        ft_network_type = config["ft_network_type"]
        assert ft_network_type in ["MLP", "LSTM"]

        # selective import
        exec(
            f"from drpo.models.base_models.encoders import FtEncoder{ft_network_type} as FtEncoder"
        )
        exec(
            f"from drpo.models.base_models.decoders import FtDecoder{ft_network_type} as FtDecoder"
        )

        self.config = config
        self.z_dim = config["z_dim"]
        self.sensors = config["sensors"]
        self.architecture = config["architecture"]
        self.use_action_in_delta = config["use_action_in_delta"]
        initialize_weights = config["initialize_weights"]

        # there must be FT sensor to provide delta_z info
        assert "ft" in self.sensors
        # there must be another sensor other than FT to provide z info
        assert len(self.sensors) > 1

        # there are multiple other sensors
        if len(self.sensors) > 2:
            self.multimodal = True
            self.num_other_sensors = len(self.sensors) - 1
            self.fusion = FusionNet(
                concat_z_dim=(len(self.sensors) - 1) * self.z_dim,
                output_z_dim=self.z_dim,
                initialize_weights=initialize_weights,
            )
        else:
            self.multimodal = False

        # the sensor should always include ft, plus other sensor such as depthmap
        for sensor in self.sensors:
            if sensor == "ft":
                params = {
                    "ft_window_size": config["ft_window_size"],
                    "use_action_in_delta": config["use_action_in_delta"],
                    "action_dim": config["action_dim"],
                }
            else:
                params = {}

            # encoder
            # TODO: add adaptibility for image and depth
            setattr(
                self,
                f"{sensor}_encoder",
                eval(f"{sensor.capitalize()}Encoder")(
                    z_dim=self.z_dim, initialize_weights=initialize_weights, **params
                ),
            )

            # decoder
            setattr(
                self,
                f"{sensor}_decoder",
                eval(f"{sensor.capitalize()}Decoder")(
                    z_dim=self.z_dim, initialize_weights=initialize_weights, **params
                ),
            )

        # TODO: try cross-entropy loss
        self.l2_loss = nn.MSELoss()

    def encode(self, obs: dict) -> dict:
        raw_encoded = {}
        for sensor in self.sensors:
            raw_encoded[sensor] = getattr(self, f"{sensor}_encoder")(obs[sensor])
        return raw_encoded

    def process_raw_encoded(self, raw_encoded: dict) -> tuple:
        # delta_z
        # NOTE: comment off to compare
        delta_z = raw_encoded["ft"]
        # z
        # TODO: test!
        z = [raw_encoded[s] for s in self.sensors if s != "ft"]
        z = torch.cat(z, dim=1)
        if self.multimodal:
            z = self.fusion(z)

        # normalization
        if self.architecture == 1:
            z = z / torch.norm(z, dim=1, keepdim=True)
        elif self.architecture == 2 or self.architecture == 3:
            delta_z = delta_z / torch.norm(delta_z, dim=1, keepdim=True)
        else:
            raise ValueError("Invalid architecture type.")

        return z, delta_z

    def decode(self, z: torch.Tensor, sensor: str) -> torch.Tensor:
        return getattr(self, f"{sensor}_decoder")(z)

    def forward(self, obs: dict) -> tuple:
        raw_encoded = self.encode(obs)
        z, delta_z = self.process_raw_encoded(raw_encoded=raw_encoded)
        encoded = (z, delta_z)
        return encoded

    def compute_loss(
        self,
        obs_prev: dict,
        encoded_prev: tuple,
        decoded_prev: dict,
        obs_curr: dict,
        encoded_curr: tuple,
        decoded_curr: dict,
    ) -> dict:
        # reconstruction loss
        recon_loss_prev = self.compute_reconstruction_loss(obs_prev, decoded_prev)
        recon_loss_curr = self.compute_reconstruction_loss(obs_curr, decoded_curr)
        recon_loss = recon_loss_prev + recon_loss_curr

        if self.architecture == 1 or self.architecture == 3:
            # temporal enforcement loss
            temp_enforce_loss = self.compute_temporal_enforcement_loss(
                encoded_prev, encoded_curr
            )

            loss = (
                self.config["reconstruction_lambda"] * recon_loss
                + self.config["temporal_enforcement_lambda"] * temp_enforce_loss
            )
            return {
                "loss": loss,
                "recon_loss": recon_loss,
                "temp_enforce_loss": temp_enforce_loss,
            }
        elif self.architecture == 2:
            loss = recon_loss
            return {
                "loss": loss,
                "recon_loss_prev": recon_loss_prev,
                "recon_loss_curr": recon_loss_curr,
            }
        else:
            raise ValueError("Invalid architecture type")

    def compute_temporal_enforcement_loss(
        self, encoded_prev: tuple, encoded_curr: tuple
    ) -> torch.Tensor:
        # architecture 2 only
        z_prev, _ = encoded_prev
        z_curr, delta_z_curr = encoded_curr

        # utilize z_curr = z_prev + delta_z_curr
        loss = self.l2_loss(z_prev + delta_z_curr, z_curr)
        return loss

    def compute_reconstruction_loss(self, obs: dict, decoded: dict) -> torch.Tensor:
        loss = 0.0
        sensors = decoded.keys()
        for sensor in sensors:
            loss += self.l2_loss(
                obs[sensor],
                decoded[sensor],
            )
        return loss
