import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse

from .blocks import pixel_up_shuffle, BasicConv2d, DWT , IWT, pixel_down_shuffle


class Pixel_Net(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None, P = 10):
        super(Pixel_Net, self).__init__()
        self.up = pixel_up_shuffle
        self.down = pixel_down_shuffle
        self.base = nn.Sequential(
            nn.Conv2d(in_channels*4, 64, kernel_size=3, stride = 1, padding = 1),
            *[BasicConv2d(64, 64, kernel_size=3, stride = 1, padding =1) for _ in range(P)],
            nn.Conv2d(64, out_channels, kernel_size=3, stride = 1, padding = 1)
        )

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        out1 = self.down(x)
        out = self.base(out1)
        out = self.up(out+out1)
        return out

class Wavelet_Net(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None, P = 10):
        super(Wavelet_Net, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels*4, 64, kernel_size=3, stride = 1, padding = 1),
            *[BasicConv2d(64, 64, kernel_size=3, stride = 1, padding =1) for _ in range(P)],
            nn.Conv2d(64, out_channels, kernel_size=3, stride = 1, padding = 1)
        )

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        out1 = self.dwt(x)
        out = self.base(out1)
        out = self.iwt(out+out1)
        return out

class DPWSDNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None, P = 10):
        super(DPWSDNet, self).__init__()
        self.WSDNet = Wavelet_Net(in_channels, out_channels, P = P)
        self.PSDNet = Pixel_Net(in_channels, out_channels, P = P)

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.001)
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        pixel_branch = self.PSDNet(x)
        wavelet_branch = self.WSDNet(x)
        out = pixel_branch + wavelet_branch
        return out


