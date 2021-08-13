import torch
import torch.nn as nn
from models.models_utils import init_weights
from models.base_models.layers import (
    conv2d,
    deconv,
)


class ProprioDecoder(nn.Module):
  def __init__(self, z_dim, initialize_weights=True):
    """
    Decodes the proprio.
    input size: n*z_dim
    """
    super().__init__()

    self.z_dim = z_dim

    self.proprio_decoder = nn.Sequential(
      nn.Linear(self.z_dim, 64),
      # nn.Dropout(0.5),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(64, 32),
      # nn.Dropout(0.5),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(32, 6),
      # nn.Dropout(0.5),
      nn.LeakyReLU(0.1, inplace=True),
      )

    if initialize_weights:
      init_weights(self.modules())

  def forward(self, z):
    """
    Predicts the proprio.

    Args:
        z: hidden state
    """
    return self.proprio_decoder(z)


class FtDecoder(nn.Module):
  def __init__(self, z_dim, initialize_weights=True):
    """
    Decodes the FT (force/torque)
    input: n*z_dim
    """
    super().__init__()
    self.z_dim = z_dim

    self.ft_decoder = nn.Sequential(
      nn.Linear(self.z_dim, 128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(128, 128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(128, 64),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(64, 48),
      nn.LeakyReLU(0.1, inplace=True),
    )

    if initialize_weights:
      init_weights(self.modules())

  def forward(self, z):
    ft = self.ft_decoder(z)
    return ft.reshape([-1, 6, 8])


class ImageDecoder(nn.Module):
  """ImageDecoder"""
  def __init__(self, z_dim, initialize_weights):
    super().__init__()
    self.z_dim = z_dim
    
    self.image_decoder = nn.Sequential(
      nn.ConvTranspose2d(self.z_dim, 128, kernel_size=2, stride=1),  # input: b*z_dim*1*1
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
    out_image = self.image_decoder(z)
    out_image = nn.functional.interpolate(
      out_image, size=(128, 128), mode="bilinear", align_corners=True)
    out_image = torch.sigmoid(out_image)
    return out_image


class DepthmapDecoder(nn.Module):
  """DepthDecoder"""
  def __init__(self, z_dim, initialize_weights):
    super().__init__()
    self.z_dim = z_dim
    
    self.depthmap_decoder = nn.Sequential(
      nn.ConvTranspose2d(self.z_dim, 128, kernel_size=2, stride=1),  # input: b*z_dim*1*1
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

    # self.depth_decoder_1 = conv2d(self.z_dim, 128, kernel_size=2, stride=1)
    # self.depth_decoder_2 = conv2d(128, 128, kernel_size=2, stride=1)
    # self.depth_decoder_3 = conv2d(128, 64, kernel_size=3, stride=1)
    # self.depth_decoder_4 = conv2d(64, 32, kernel_size=5, stride=1)
    # self.depth_decoder_5 = conv2d(32, 16, kernel_size=3, stride=1)
    # self.depth_decoder_6 = conv2d(16, 1, kernel_size=2, stride=1)    

    if initialize_weights:
      init_weights(self.modules())

  def forward(self, z):
    # z: n*z_dim
    z = z.unsqueeze(2).unsqueeze(3)
    out_image = self.depthmap_decoder(z)
    # out_image = z
    # for i in range(1, 7):
    #   out_image = nn.functional.interpolate(out_image, scale_factor=2, mode="bilinear")
    #   out_image = eval("self.depth_decoder_{}".format(i))(out_image)
    # out_image = self.image_decoder(z)
    out_image = nn.functional.interpolate(out_image, size=(128, 128), mode="bilinear", align_corners=True)
    return out_image
