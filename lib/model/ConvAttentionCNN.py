import torch.nn as nn
from .blocks import ConvAttentionBlock, ConvAttentionResidualBlock


class ConvAttentionCNN(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(ConvAttentionCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=2, depth=1),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=2, depth=1),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU(),
            ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=2, depth=1),
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


class TemporalConvAttentionCNN(nn.Module):
    def __init__(self, in_channels, out_channels, tseq_len = 1, conv_block=None):
        super(TemporalConvAttentionCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        # self.conv_attention1 = ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=1, depth=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=7, padding=3)
        self.conv_attention2 = ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=1, depth=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)
        # self.conv_attention3 = ConvAttentionBlock(kernel_size=3, stride=1, padding=1, conv_block=None, layer_depth=1, depth=1)
        self.conv4 = nn.Conv2d(16 * tseq_len, out_channels, kernel_size=5, padding=2)
 
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        # B, T, C, H, W = x.shape
        # r_x = x.reshape(-1, C, H, W)
        # out1 = self.conv1(r_x)
        # r_x = out1.reshape(B,-1, H, W)
        # att_out1 = self.conv_attention1(r_x)

        # r_x = att_out1.reshape(B*T, -1, H, W)
        # out2 = self.conv2(r_x)
        # r_x = out2.reshape(B,-1, H,W)
        # att_out2 = self.conv_attention2(r_x)

        # r_x = att_out2.reshape(B*T, -1 , H, W)
        # out3 = self.conv3(r_x)
        # r_x = out3.reshape(B,-1, H, W)
        # att_out3 = self.conv_attention3(r_x)

        # # r_x = att_out3.reshape(B,-1,H,W)
        # out = self.conv4(att_out3)

        B, T, C, H, W = x.shape
        r_x = x.reshape(-1, C, H, W)
        out1 = self.conv1(r_x)
        # r_x = out1.reshape(B,-1, H, W)
        # att_out1 = self.conv_attention1(r_x)

        # r_x = att_out1.reshape(B*T, -1, H, W)
        out2 = self.conv2(out1)
        r_x = out2.reshape(B,-1, H,W)
        att_out2 = self.conv_attention2(r_x)

        r_x = att_out2.reshape(B*T, -1 , H, W)
        out3 = self.conv3(r_x)
        r_x = out3.reshape(B,-1, H, W)
        # att_out3 = self.conv_attention3(r_x)

        # r_x = att_out3.reshape(B,-1,H,W)
        out = self.conv4(r_x)
        return out


