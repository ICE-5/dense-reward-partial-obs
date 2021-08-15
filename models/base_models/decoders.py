import torch
import torch.nn as nn
from models.models_utils import init_weights
from models.base_models.layers import (
    conv2d,
    deconv,
)


class FtDecoderMLP(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    FT (force/torque) decoder
    input: n*z_dim
    """
        super().__init__()
        self.z_dim = z_dim

        # adapt to different window size, by default use 8
        if "ft_window_size" in kwargs.keys():
            self.ft_window_size = kwargs["ft_window_size"]
        else:
            self.ft_window_size = 8

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 6 * self.ft_window_size),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, delta_z):
        out = self.decoder(delta_z)
        return out.reshape([-1, 6, self.ft_window_size])


# TODO: craft and tune LSTM decoder
class FtDecoderLSTM(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    FT (force/torque) decoder
    input: n*z_dim
    """
        super().__init__()
        self.z_dim = z_dim

        # adapt to different window size, by default use 8
        if "ft_window_size" in kwargs.keys():
            self.ft_window_size = kwargs["ft_window_size"]
        else:
            self.ft_window_size = 8

        self.decoder = nn.LSTM(input_size=self.z_dim, hidden_size=32, batch_first=True,)
        self.dense = nn.Parameter(
            torch.randn((1, 32, 6), dtype=torch.double), requires_grad=True
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, self.ft_window_size, 1)
        # shape efore piping into LSTM: [batch_size, ft_window_size, z_dim]
        out, _ = self.decoder(z)
        out = torch.matmul(out, self.dense)
        # output should be [batch_size, 6]
        return out.reshape([-1, 6, self.ft_window_size])


class ImageDecoder(nn.Module):
    """Image decoder"""

    def __init__(self, z_dim, initialize_weights, **kwargs):
        super().__init__()
        self.z_dim = z_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.z_dim, 128, kernel_size=2, stride=1
            ),  # input: b*z_dim*1*1
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),  # input: b*z_dim*2*2
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),  # input: b*z_dim*4*4
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1),  # input: b*z_dim*8*8
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),  # input: b*z_dim*16*16
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2),  # input: b*z_dim*32*32
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1),  # input: b*z_dim*32*32
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, z):
        # z: n*z_dim
        z = z.unsqueeze(2).unsqueeze(3)
        out = self.decoder(z)
        out = nn.functional.interpolate(
            out, size=(128, 128), mode="bilinear", align_corners=True
        )
        out = torch.sigmoid(out)
        return out


class DepthmapDecoder(nn.Module):
    """Depthmap decoder"""

    def __init__(self, z_dim, initialize_weights, **kwargs):
        super().__init__()
        self.z_dim = z_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.z_dim, 128, kernel_size=2, stride=1
            ),  # input: b*z_dim*1*1
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  # input: b*z_dim*2*2
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # input: b*z_dim*4*4
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # input: b*z_dim*8*8
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # input: b*z_dim*16*16
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # input: b*z_dim*32*32
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, z):
        # z: n*z_dim
        z = z.unsqueeze(2).unsqueeze(3)
        out = self.decoder(z)
        out = nn.functional.interpolate(
            out, size=(128, 128), mode="bilinear", align_corners=True
        )
        return out


# class ProprioDecoder(nn.Module):
#     def __init__(self, z_dim, initialize_weights=True):
#         """
#     Decodes the proprio.
#     input size: n*z_dim
#     """
#         super().__init__()

#         self.z_dim = z_dim

#         self.proprio_decoder = nn.Sequential(
#             nn.Linear(self.z_dim, 64),
#             # nn.Dropout(0.5),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(64, 32),
#             # nn.Dropout(0.5),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(32, 6),
#             # nn.Dropout(0.5),
#             nn.LeakyReLU(0.1, inplace=True),
#         )

#         if initialize_weights:
#             init_weights(self.modules())

#     def forward(self, z):
#         """
#     Predicts the proprio.

#     Args:
#         z: hidden state
#     """
#         return self.proprio_decoder(z)
