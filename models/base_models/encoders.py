import torch
import torch.nn as nn
from torch.nn.modules import dropout
from models.models_utils import init_weights
from models.base_models.layers import CausalConv1D, Flatten, conv2d


class FtEncoderMLP(nn.Module):
    def __init__(self, z_dim, initailize_weights=True, **kwargs):
        """
    FT (force/torque) encoder taken from selfsupervised code
    Input size: [batch_size, 6, ft_window_size]
    """
        super().__init__()
        self.z_dim = z_dim

        # adapt to different window size, by default use 8
        if "ft_window_size" in kwargs.keys():
            self.ft_window_size = kwargs["ft_window_size"]
        else:
            raise ValueError("please specify FT window size in config")

        self.encoder = nn.Sequential(
            nn.Linear(6 * self.ft_window_size, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, x):
        x = x.reshape([-1, 6 * self.ft_window_size])
        return self.encoder(x)


# TODO: craft and tune LSTM encoder
class FtEncoderLSTM(nn.Module):
    def __init__(self, z_dim, initailize_weights=True, **kwargs):
        """
    FT (force/torque) encoder taken from selfsupervised code
    Input size: [batch_size, 6, ft_window_size]
    """
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.LSTM(
            input_size=6,
            hidden_size=self.z_dim,
            batch_first=True,
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        # shape efore piping into LSTM: [batch_size, ft_window_size, 6]
        _, (out, _) = self.encoder(x)
        # output should be [batch_size, z_dim]
        return out.squeeze()


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True, **kwargs):
        """
    Image encoder taken from Making Sense of Vision and Touch
    Input size: [batch_size, 128, 128, 3]
    """
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            conv2d(3, 16, kernel_size=7, stride=2),
            conv2d(16, 32, kernel_size=5, stride=2),
            conv2d(32, 64, kernel_size=5, stride=2),
            conv2d(64, 128, stride=2),
            conv2d(128, 128, stride=2),
            conv2d(128, self.z_dim, stride=2),
            Flatten(),
            nn.Linear(4 * self.z_dim, self.z_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, x):
        out = self.encoder(x)
        return out


class DepthmapEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True, **kwargs):
        """
    Simplified Depthmap Encoder taken from Making Sense of Vision and Touch
    Input size: [batch_size, 128, 128, 1]
    """
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            conv2d(1, 32, kernel_size=3, stride=2),
            conv2d(32, 64, kernel_size=3, stride=2),
            conv2d(64, 64, kernel_size=4, stride=2),
            conv2d(64, 64, stride=2),
            conv2d(64, 128, stride=2),
            conv2d(128, self.z_dim, stride=2),
            Flatten(),
            nn.Linear(4 * self.z_dim, self.z_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, x):
        out = self.encoder(x)
        return out


# class ProprioEncoder(nn.Module):
#     def __init__(self, z_dim, initailize_weights=True):
#         """
#     Proprio encoder taken from selfsupervised code
#     input size: n*12
#     """
#         super().__init__()
#         self.z_dim = z_dim

#         self.encoder = nn.Sequential(
#             nn.Linear(6, 32),
#             # nn.Dropout(0.5),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(32, 64),
#             # nn.Dropout(0.5),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(64, 64),
#             # nn.Dropout(0.5),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(64, self.z_dim),
#             # nn.Dropout(0.5),
#             nn.LeakyReLU(0.1, inplace=True),
#         )

#         if initailize_weights:
#             init_weights(self.modules())

#     def forward(self, x):
#         return self.encoder(x)


class FusionNet(nn.Module):
    def __init__(self, concat_z_dim, output_z_dim, initailize_weights=True):
        super().__init__()

        self.fusion_net = nn.Sequential(
            nn.Linear(concat_z_dim, output_z_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(output_z_dim, output_z_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, x):
        return self.fusion_net(x)

