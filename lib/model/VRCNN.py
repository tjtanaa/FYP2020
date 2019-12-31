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

class VRCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(ARCNN, self).__init__()
        # print("in_channels, ", in_channels, "\t", "channel_size: ",out_channels)
        if conv_block is None:
            conv_block = PreluConv2d        
        self.layer1 = conv_block(in_channels, 64, kernel_size=5, stride = 1, padding = 2)
        self.layer2_1 = conv_block(64, 16, kernel_size=5, stride = 1, padding = 2)
        self.layer2_2 = conv_block(64, 32, kernel_size=3, stride = 1, padding = 1)
        self.layer3_1 = conv_block(48, 16, kernel_size=3, stride = 1, padding = 1)
        self.layer3_2 = conv_block(48, 32, kernel_size=1, stride = 1, padding = 0)
        self.layer4 = conv_block(48, out_channels, kernel_size=3, stride = 1, padding = 1)
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        """
            Input:
            x : [1 x N x C x H x W ]
    
        """
        cur_patch = x[0]
        # print("pre_patch size, ", pre_patch.shape)


        out = self.layer1(cur_patch)
        out1 = self.layer2_1(out)
        out2 = self.layer2_2(out)
        outputs = [out1, out2]
        concat1 = torch.cat(outputs, 1)

        out1 = self.layer3_1(concat1)
        out2 = self.layer3_2(concat1)
        outputs = [out1, out2]
        concat2 = torch.cat(outputs, 1)

        out = self.layer4(concat2)

        return out
