import torch.nn as nn
from .blocks import ConvAttentionBlock, ConvAttentionResidualBlock


class ConvAttentionCNN(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(ConvAttentionCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=1, depth=3),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=1, depth=3),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU(),
            ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=1, depth=3),
        )
        self.last = nn.Conv2d(16, out_channels, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


