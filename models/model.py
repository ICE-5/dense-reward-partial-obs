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

        self.concat_z_dim = 0
        for sensor in config["sensor_used_in_model"]:
            setattr(self, f"{sensor}_z_dim", config[f"{sensor}_z_dim"])
            setattr(
                self,
                f"{sensor}_encoder",
                eval(f"{sensor.capitalize()}Encoder")(
                    eval(f"self.{sensor}_z_dim"), initialize_weights
                ),
            )
            self.concat_z_dim += config[f"{sensor}_z_dim"]
            setattr(
                self,
                f"{sensor}_decoder",
                eval(f"{sensor.capitalize()}Decoder")(self.z_dim, initialize_weights),
            )

        # init fusion net
        self.fusion_net = FusionNet(self.concat_z_dim, self.z_dim, initialize_weights)

        # init loss functions
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def bce_with_continuous_target(self, input, target):
        eplison = 1e-6
        return self.kl_loss(
            torch.log(input + eplison), target + eplison
        ) + self.kl_loss(torch.log(1.0 - input + eplison), (1.0 - target + eplison))

    def encode(self, obs):
        # encode multi-modal sensor data into a hidden vector
        for sensor in self.config["sensor_used_in_model"]:
            # TODO: try locals() or globals()
            input = obs[sensor].double()
            if len(input.shape) == 2:
                input = torch.unsqueeze(input, 0)

            setattr(self, f"{sensor}_enc", eval(f"self.{sensor}_encoder")(input))
            if "map" in sensor or "img" in sensor:
                setattr(self, f"{sensor}_enc", getattr(self, f"{sensor}_enc")[0])

        # fuse the 4 hidden vectors
        concat_encs = []
        for sensor in self.config["sensor_used_in_model"]:
            # print(f"{sensor}, {eval(f'{sensor}_enc').shape}")
            concat_encs.append(getattr(self, f"{sensor}_enc"))
        z = self.fusion_net(torch.cat(concat_encs, dim=1))
        z = z / torch.norm(z, dim=1, keepdim=True)  # normalize the z
        return z

    def decode(self, z):
        decoded_output = {}
        for sensor in self.config["sensor_used_in_model"]:
            decoded_output[f"decoded_{sensor}"] = eval(f"self.{sensor}_decoder")(z)
        return decoded_output

    def forward(self, x):
        z = self.encode(x)
        decoded_output = self.decode(z)
        return z, decoded_output

    def compute_loss(self, a, decoded_a, z_a, b, decoded_b, z_b, z_g):
        # loss should be based on: recon_loss + cmp_loss
        recon_a_loss = self.compute_reconstruction_loss(a, decoded_a)
        recon_b_loss = self.compute_reconstruction_loss(b, decoded_b)

        # cmp_loss = self.compute_comparison_loss(z_a, z_b, z_g, a["depth"], b["depth"])
        cmp_loss = self.compute_cos_comparison_loss(
            z_a, z_b, z_g, a["depth"], b["depth"]
        )
        # out_cone_loss = self.compute_out_cone_loss(z_a, z_b, z_g, z_r)

        recon_loss = recon_a_loss + recon_b_loss
        all_loss = (
            self.config["reconstruction_lambda"] * recon_loss
            + self.config["comparison_lambda"] * cmp_loss
        )
        return all_loss, recon_loss, cmp_loss

    def compute_reconstruction_loss(self, data_input, decoded_output):
        recon_loss = 0.0
        for sensor in self.config["sensor_used_in_model"]:
            # if "map" in sensor or "img" in sensor:
            #     recon_loss += self.bce_with_continuous_target(
            #         decoded_output[f"decoded_{sensor}"], data_input[f"{sensor}"],
            #     )
            # else:
            #     recon_loss += self.l2_loss(
            #         data_input[f"{sensor}"], decoded_output[f"decoded_{sensor}"],
            #     )
            recon_loss += self.l2_loss(
                data_input[f"{sensor}"], decoded_output[f"decoded_{sensor}"],
            )
        return recon_loss

    def compute_comparison_loss(self, z_a, z_b, z_g, depth_a, depth_b):
        dist_ag = torch.norm(z_a - z_g, dim=1)
        dist_bg = torch.norm(z_b - z_g, dim=1)
        if_within = torch.abs(depth_a - depth_b) <= self.config["depth_margin"]
        relative_dist_ab = (dist_bg - dist_ag) * ((depth_b > depth_a) * 2.0 - 1.0)
        # relative_dist_ab = (dist_bg - dist_ag) * ((depth_b > depth_a))
        within_loss = torch.max(
            torch.zeros_like(dist_ag), -1.0 * relative_dist_ab
        ) + torch.max(
            torch.zeros_like(dist_ag),
            relative_dist_ab - self.config["within_threshold"],
        )
        outside_loss = torch.max(
            torch.zeros_like(dist_ag),
            self.config["outside_threshold"] - relative_dist_ab,
        )
        comparison_loss = (
            if_within.float() * within_loss + (1 - if_within.float()) * outside_loss
        )

        # import pdb; pdb.set_trace()
        return comparison_loss.mean()

    def compute_cos_comparison_loss(self, z_a, z_b, z_g, depth_a, depth_b):
        dist_ag = 1.0 - self.compute_cos_distance(z_a, z_g)
        dist_bg = 1.0 - self.compute_cos_distance(z_b, z_g)
        if_within = torch.abs(depth_a - depth_b) <= self.config["depth_margin"]
        relative_dist_ab = (dist_bg - dist_ag) * ((depth_b > depth_a) * 2.0 - 1.0)
        # relative_dist_ab = (dist_bg - dist_ag) * ((depth_b > depth_a))
        within_loss = torch.max(
            torch.zeros_like(dist_ag), -1.0 * relative_dist_ab
        ) + torch.max(
            torch.zeros_like(dist_ag),
            relative_dist_ab - self.config["within_threshold"],
        )
        outside_loss = torch.max(
            torch.zeros_like(dist_ag),
            (torch.abs(depth_b - depth_a) // self.config["depth_margin"]).float()
            * self.config["outside_threshold"]
            - relative_dist_ab,
        )
        # outside_loss = torch.max(torch.zeros_like(dist_ag), \
        #   self.config["outside_threshold"] - relative_dist_ab)
        comparison_loss = (
            if_within.float() * within_loss + (1 - if_within.float()) * outside_loss
        )

        return comparison_loss.mean()

    def compute_out_cone_loss(self, z_a, z_b, z_g, z_r=None):
        # z_r is the hidden state of random observation
        # if z_r exists, the angle between ag and rg should be larger than certain thresh
        z_a = z_a.unsqueeze(dim=1)
        z_b = z_b.unsqueeze(dim=1)
        z_g = z_g.unsqueeze(dim=2)

        dot_ag = torch.matmul(z_a, z_g).squeeze()
        dot_bg = torch.matmul(z_b, z_g).squeeze()
        # the dot product should be larger than cos(cone_angle_threshold)
        dot_product_thresh = np.cos(self.config["cone_angle_threshold"]) + 0.01

        out_cone_loss = torch.max(
            torch.zeros_like(dot_ag), dot_product_thresh - dot_ag
        ) + torch.max(torch.zeros_like(dot_bg), dot_product_thresh - dot_bg)

        # dist_ag = torch.norm(z_a-z_g, dim=1)
        # ag_upper_bound = 2 * 1. * np.sin(self.config["cone_angle_threshold"]/2.)
        # out_cone_loss += torch.max(torch.zeros_like(dist_ag), dist_ag - ag_upper_bound)

        # check if z_r exists
        if z_r is not None:
            z_r = z_r.unsqueeze(dim=1)
            dot_rg = torch.matmul(z_r, z_g).squeeze()
            out_cone_loss += torch.max(
                torch.zeros_like(dot_rg), dot_rg - dot_product_thresh
            )

        return out_cone_loss.mean()

    def compute_cos_distance(self, z_a, z_b):
        # compute the cos distance of two hidden vectors, z_a and z_b are already normalized
        # z_a: [b, h_dim], z_b: [b, h_dim]
        z_a = z_a.unsqueeze(dim=1)
        z_b = z_b.unsqueeze(dim=2)
        res = torch.matmul(z_a, z_b).squeeze()
        return res

