import torch
import torch.nn as nn
import torch.nn.functional as F

class inception_block(torch.nn.Module):

    def __init__(self, in_channels, pool_features=None, conv_block=None):
        super(inception_block, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 8, kernel_size=1, stride = 1, padding =0)

        self.branch3x3_1 = conv_block(in_channels, 32, kernel_size=1, stride = 1, padding = 0)
        self.branch3x3_2 = conv_block(32, 32, kernel_size=3, stride = 1, padding = 1)

        self.branch5x5_1 = conv_block(in_channels, 16, kernel_size=1, stride = 1, padding = 0)
        self.branch5x5_2 = conv_block(16, 16, kernel_size=5, stride = 1, padding = 2)

        self.branch7x7_1 = conv_block(in_channels, 8, kernel_size=1, stride = 1, padding = 0)
        self.branch7x7_2 = conv_block(8, 8, kernel_size=7, stride = 1, padding = 3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)

        outputs = [branch1x1, branch3x3, branch5x5, branch7x7]

        # for i in range(len(outputs)):
        #     print("i: ", i ," \t shape: ", outputs[i].shape)
        # # print()
        # print(torch.cat(outputs, 1).shape)
        # exit()

        return torch.cat(outputs, 1)


class BasicConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)