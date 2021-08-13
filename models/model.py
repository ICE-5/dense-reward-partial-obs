import torch
import torch.nn as nn
import numpy as np

from models.base_models.encoders import *
from models.base_models.decoders import *


class PartialObsAutoEncoder(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.config = config
        self.z_dim = config["z_dim"]
        initialize_weights = config["initialize_weights"]

        # let's assume we are using both ft and depthmap
        self.ft_encoder = FtEncoder(self.z_dim, initialize_weights)
        self.depthmap_encoder = DepthmapEncoder(self.z_dim, initialize_weights)

        self.ft_decoder = FtDecoder(self.z_dim, initialize_weights)
        self.depthmap_decoder = DepthmapDecoder(self.z_dim, initialize_weights)

        # init loss functions
        self.l2_loss = nn.MSELoss()

    def encode(self, obs):
        ft = obs["ft"].double()
        if len(ft.shape) == 2:
            ft = torch.unsqueeze(ft, 0)
        delta_z = self.ft_encoder(ft)

        depthmap = obs["depthmap"].double()
        if len(depthmap.shape) == 2:
            depthmap = torch.unsqueeze(depthmap, 0)
        z, _ = self.depthmap_encoder(depthmap)
        z = z / torch.norm(z, dim=1, keepdim=True)

        return z, delta_z

    def decode(self, z, delta_z):
        decoded_output = {}
        decoded_output["ft"] = self.ft_decoder(delta_z)
        decoded_output["depthmap"] = self.depthmap_decoder(z)
        return decoded_output

    def forward(self, x):
        z, delta_z = self.encode(x)
        decoded_output = self.decode(z, delta_z)
        return z, delta_z, decoded_output

    def compute_loss(
        self, a, decoded_a, z_a, delta_z_a, b, decoded_b, z_b, delta_z_b, 
    ):
        # reconstruction loss
        recon_a_loss = self.compute_reconstruction_loss(a, decoded_a)
        recon_b_loss = self.compute_reconstruction_loss(b, decoded_b)

        # temporal loss
        temporal_loss = self.l2_loss(z_b, z_a + delta_z_b)

        recon_loss = recon_a_loss + recon_b_loss
        loss = (
            self.config["reconstruction_lambda"] * recon_loss
            + self.config["comparison_lambda"] * temporal_loss
        )
        return loss, recon_loss, temporal_loss

    def compute_reconstruction_loss(self, data_input, decoded_output):
        recon_loss = 0.0
        for sensor in self.config["sensor_used_in_model"]:
            recon_loss += self.l2_loss(
                data_input[f"{sensor}"], decoded_output[f"{sensor}"],
            )
        return recon_loss

    def compute_cos_distance(self, z_a, z_b):
        # compute the cos distance of two hidden vectors, z_a and z_b are already normalized
        # z_a: [b, h_dim], z_b: [b, h_dim]
        z_a = z_a.unsqueeze(dim=1)
        z_b = z_b.unsqueeze(dim=2)
        res = torch.matmul(z_a, z_b).squeeze()
        return res

