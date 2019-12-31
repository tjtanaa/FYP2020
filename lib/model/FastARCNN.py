import torch
import torch.nn.functional as F
import torch.nn as nn

class PreluConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PreluConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.prelu_l = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.prelu_l(x)

class FastARCNN(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(FastARCNN, self).__init__()
        if conv_block is None:
            conv_block = PreluConv2d
        self.feature_extraction = conv_block(in_channels, 64, kernel_size=9, stride = 2, padding = 4)
        self.mapping1 = conv_block(64, 32, kernel_size=1, stride = 1, padding = 0)
        self.feature_enhancement = conv_block(32, 32, kernel_size=7, stride = 1, padding = 3)
        self.mapping2 = conv_block(32, 64, kernel_size=1, stride = 1, padding = 0)
        self.reconstruction = nn.ConvTranspose2d(64, 3, kernel_size=9, stride=2, padding=4, output_padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):

        cur_patch = x[1]
        fex = self.feature_extraction(cur_patch)
        fmp1 = self.mapping1(fex)
        fen = self.feature_enhancement(fmp1)
        fmp2 = self.mapping2(fen)
        frc = self.reconstruction(fmp2)

        return frc