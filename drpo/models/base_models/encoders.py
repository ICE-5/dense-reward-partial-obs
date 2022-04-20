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
            self.ft_window_size = kwargs["ft_window_size"]
            use_action_in_delta = kwargs["use_action_in_delta"]
            action_dim = kwargs["action_dim"]
        except:
            raise IOError("Missing essential arguments.")

        if use_action_in_delta:
            out_dim = z_dim - action_dim
        else:
            out_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(6 * self.ft_window_size, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, out_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        x = x.reshape([-1, 6 * self.ft_window_size])
        return self.encoder(x)


# TODO: craft and tune LSTM encoder
class FtEncoderLSTM(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
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

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        # x = torch.transpose(x, 1, 2)
        # shape before piping into LSTM: [batch_size, ft_window_size, 6]
        _, (out, _) = self.encoder(x)
        # output should be [batch_size, z_dim]
        return out.squeeze(0)


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    Image encoder taken from Making Sense of Vision and Touch
    Input size: [batch_size, num_channels, H, W]
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

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        out = self.encoder(x)
        return out


class DepthmapEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True, **kwargs):
        """
    Simplified Depthmap Encoder taken from Making Sense of Vision and Touch
    Input size: [batch_size, num_channels, H, W]
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

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, x):
        out = self.encoder(x)
        return out


class ProprioEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True):
        """
    Proprio encoder taken from selfsupervised code
    Input size: [batch_size, 32]
    """
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(32, 32),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, self.z_dim),
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

