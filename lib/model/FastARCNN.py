
import torch.nn as nn

class FastARCNN(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(FastARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.ConvTranspose2d(64, out_channels, kernel_size=9, stride=2, padding=4, output_padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x