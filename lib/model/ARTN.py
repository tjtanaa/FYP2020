import torch
import torch.nn.functional as F
from .inception_block import inception_block, BasicConv2d

class temporal_branch(torch.nn.Module):

    def __init__(self, in_channels, channel_size, conv_block=None):
        super(temporal_branch, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv1 = conv_block(in_channels, channel_size, kernel_size=7, stride = 1, padding =3)
        self.conv2 = conv_block(in_channels, channel_size, kernel_size=1, stride = 1, padding =0)
        self.conv3 = conv_block(in_channels, channel_size, kernel_size=3, stride = 1, padding =1)

    def forward(self, x):
    	c1 = F.relu(self.conv1(x))
    	c2 = F.relu(self.conv2(x))
    	output = F.relu(self.conv3(x))
    	return output


class ARTN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(ARTN, self).__init__()
        # print("in_channels, ", in_channels, "\t", "channel_size: ",out_channels)
        self.pre_branch = temporal_branch(in_channels, 32)
        self.cur_branch = temporal_branch(in_channels, 64)
        self.post_branch = temporal_branch(in_channels, 32)
        self.inception_block1 = inception_block(128)
        self.inception_block2 = inception_block(64)
        self.conv5 = BasicConv2d(64, out_channels, kernel_size=5, stride = 1, padding =2)

    def forward(self, x):
    	"""
			Input:
			x : [T x N x C x H x W ]
	
    	"""
    	pre_patch = x[0]
    	cur_patch = x[1]
    	post_patch = x[2]
    	# print("pre_patch size, ", pre_patch.shape)

    	pre_output = self.pre_branch(pre_patch)		# [1 x N x C x H x W]
    	cur_output = self.cur_branch(cur_patch)
    	post_output = self.post_branch(post_patch)

    	outputs = [pre_output, cur_output, post_output]

    	concat1 = torch.cat(outputs, 1)
    	# print(concat1.shape)
    	# exit()

    	incep1 = self.inception_block1(concat1)
    	# print(incep1.shape)
    	# exit()    	
    	incep2 = self.inception_block2(incep1)
    	output_patch = self.conv5(incep2)

    	return output_patch

