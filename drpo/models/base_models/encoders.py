import torch
import torch.nn as nn
from torch.nn.modules import dropout

from drpo.models.models_utils import init_weights
from drpo.models.base_models.layers import CausalConv1D, Flatten, conv2d


class FtEncoderMLP(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    FT (force/torque) encoder taken from selfsupervised code
    Input size: [batch_size, 6, ft_window_size]
    """
        super().__init__()

        try:
            if kwargs["use_action_in_delta"]:
                in_dim = 6 * kwargs["ft_window_size"] + kwargs["action_dim"]
            else:
                in_dim = 6 * kwargs["ft_window_size"]
        except KeyError:
            raise IOError("Missing essential arguments.")

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        return self.encoder(x)


# class FtEncoderLSTM(nn.Module):
#     def __init__(self, z_dim, initialize_weights=True, **kwargs):
#         """
#     FT (force/torque) encoder taken from selfsupervised code
#     Input size: [batch_size, 6, ft_window_size]
#     """
#         super().__init__()

#         self.encoder = nn.LSTM(
#             input_size=6,
#             hidden_size=z_dim,
#             batch_first=True,
#         )

#         if initialize_weights:
#             init_weights(self.modules())

#     def forward(self, x):
#         # x = torch.transpose(x, 1, 2)
#         # shape before piping into LSTM: [batch_size, ft_window_size, 6]
#         _, (out, _) = self.encoder(x)
#         # output should be [batch_size, z_dim]
#         return out.squeeze(0)


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    Image encoder taken from Making Sense of Vision and Touch
    Input size: [batch_size, num_channels, image_dim, image_dim]
    """
        super().__init__()

        self.encoder = nn.Sequential(
            conv2d(3, 16, kernel_size=7, stride=2),
            conv2d(16, 32, kernel_size=5, stride=2),
            conv2d(32, 64, kernel_size=5, stride=2),
            conv2d(64, 128, stride=2),
            conv2d(128, 128, stride=2),
            conv2d(128, z_dim, stride=2),
            Flatten(),
            nn.Linear(4 * z_dim, z_dim),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        out = self.encoder(x)
        return out


class DepthEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    Simplified Depthmap Encoder taken from Making Sense of Vision and Touch
    Input size: [batch_size, num_channels, image_dim, image_dim]
    """
        super().__init__()

        self.encoder = nn.Sequential(
            conv2d(1, 32, kernel_size=3, stride=2),
            conv2d(32, 64, kernel_size=3, stride=2),
            conv2d(64, 64, kernel_size=4, stride=2),
            conv2d(64, 64, stride=2),
            conv2d(64, 128, stride=2),
            conv2d(128, z_dim, stride=2),
            Flatten(),
            nn.Linear(4 * z_dim, z_dim),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        out = self.encoder(x)
        return out


class ProprioEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    Proprio encoder taken from selfsupervised code
    Input size: [batch_size, proprio_dim] or [batch_size, proprio_dim+object_dim]
    """
        super().__init__()

        try:
            if kwargs["use_object_in_proprio"]:
                in_dim = kwargs["proprio_dim"] + kwargs["object_dim"]
            else:
                in_dim = kwargs["proprio_dim"]
        
        except KeyError:
            raise IOError("Missing essential arguments.")

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, z_dim),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        return self.encoder(x)


class FusionNet(nn.Module):
    def __init__(self, concat_z_dim, output_z_dim, initialize_weights=True):
        super().__init__()

        self.fusion_net = nn.Sequential(
            nn.Linear(concat_z_dim, output_z_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(output_z_dim, output_z_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        return self.fusion_net(x)

