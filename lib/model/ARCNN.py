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

class ARCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(ARCNN, self).__init__()
        # print("in_channels, ", in_channels, "\t", "channel_size: ",out_channels)
        if conv_block is None:
            conv_block = PreluConv2d        
        self.feature_extraction = conv_block(in_channels, 64, kernel_size=9, stride = 1, padding = 4)

        self.feature_enhancement = conv_block(64, 32, kernel_size=7, stride = 1, padding = 3)
        self.mapping = conv_block(32, 16, kernel_size=1, stride = 1, padding = 0)
        self.reconstruction = nn.Conv2d(16, 3, kernel_size=5, padding=2) # no batch normalization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        """
            Input:
            x : [1 x N x C x H x W ]
    
        """
        cur_patch = x[0]
        # print("pre_patch size, ", pre_patch.shape)

        fex= self.feature_extraction(cur_patch)
        fen= self.feature_enhancement(fex)
        fmp= self.mapping(fen)
        frc= self.reconstruction(fmp)

        return frc



