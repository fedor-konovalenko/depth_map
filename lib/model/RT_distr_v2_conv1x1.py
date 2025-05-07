import torch
from torch import nn
import torch.nn.functional as F

def ConvBlock_d(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True)
  )

class UpConvBlock_d(nn.Module):
  def __init__(self, in_channels, out_channels, scale):
    super(UpConvBlock_d, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.scale = scale

  def forward(self, x):
    x = self.conv(x)
    x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
    return x

def ConvBlock(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True)
  )

class UpConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UpConvBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.conv(x)
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    return x

class ShiftedELU(nn.Module):
  def __init__(self, shift=1.0):
    super(ShiftedELU, self).__init__()
    self.shift = shift
    self.elu = nn.ELU()

  def forward(self, x):
    return self.elu(x) + self.shift

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = ShiftedELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x

class RT_MonoDepth_Mk2(nn.Module):

  def __init__(self, decode_distr):
    super(RT_MonoDepth_Mk2, self).__init__()
    # Pyramid Encoder
    self.conv1 = ConvBlock(3, 32)
    self.conv2 = ConvBlock(32, 64)
    self.conv3 = ConvBlock(64, 128)
    self.conv4 = ConvBlock(128, 256)
    self.conv5 = ConvBlock_d(256, 256)
    self.conv6 = ConvBlock_d(256, 256)
    self.fc = nn.Linear(256, 256)
    self.a1 = nn.PReLU()
    self.decode_distr = decode_distr

    # Depth Decoder
    self.upconv5 = UpConvBlock_d(256, 256, 4)
    self.upconv4 = UpConvBlock_d(256, 256, 4)
    self.upconv3 = UpConvBlock(256, 128)
    self.decoder3 = Decoder(128)
    self.upconv2 = UpConvBlock(128, 64)
    self.decoder2 = Decoder(64)
    self.upconv1 = UpConvBlock(64, 32)
    self.decoder1 = Decoder(64)
    self.upconv0 = UpConvBlock(64, 32)
    self.decoder0 = Decoder(32)

  def forward(self, x):
    # Pyramid Encoder
    F1 = self.conv1(x)
    F2 = self.conv2(F1)
    F3 = self.conv3(F2)
    F4 = self.conv4(F3)
    F5 = self.conv5(F4)
    F6 = self.conv6(F5)
    F7 = torch.flatten(F6, start_dim=1)
    DD = self.a1(self.fc(F7))
    if self.decode_distr:
      F6 = (DD.unsqueeze(-1)).unsqueeze(-1)

    # Depth Decoder
    B5 = self.upconv4(F6)
    B4 = self.upconv4(B5)
    B3 = self.upconv3(F4)
    B3 = B3 + F3 # Fusion Block 3
    B2 = self.upconv2(B3)
    B2 = B2 + F2 # Fusion Block 2
    B1 = self.upconv1(B2)
    B1 = torch.cat((B1, F1), dim=1) # Fusion Block 1
    B0 = self.upconv0(B1)

    D0 = self.decoder0(B0)

    if self.training:
      D1 = self.decoder1(B1)
      D2 = self.decoder2(B2)
      D3 = self.decoder3(B3)
      return D0, D1, D2, D3, DD
    else:
      return D0