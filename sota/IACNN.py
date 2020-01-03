import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
  """
  Building block of other networks, credited to
  https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
  """
  def __init__(self, in_channels, out_channels, **kwargs):
    super(BasicConv2d, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return F.relu(x, inplace=True)


class Inception(nn.Module):
  """
  Inception module modified in "A Pseudo-Blind Convolutional Neural Network
  for the Reduction of Compression Artifacts"
  """
  def __init__(self, in_channels):
    # padding?
    super(Inception, self).__init__()
    self.branch1x1_a = BasicConv2d(in_channels, 8, kernel_size=1)

    self.branch1x1_b = BasicConv2d(in_channels, 32, kernel_size=1)
    self.branch3x3_b = BasicConv2d(32, 32, kernel_size=3)

    self.branch1x1_c = BasicConv2d(in_channels, 16, kernel_size=1)
    self.branch5x5_c = BasicConv2d(16, 16, kernel_size=5)

    self.branch1x1_d = BasicConv2d(in_channels, 8, kernel_size=1)
    self.branch7x7_d = BasicConv2d(8, 8, kernel_size=7)

  def forward(self, x):
    branch_a = self.branch1x1_a(x)

    branch_b = self.branch1x1_b(x)
    branch_b = self.branch3x3_b(branch_b)

    branch_c = self.branch1x1_c(x)
    branch_c = self.branch5x5_c(branch_c)

    branch_d = self.branch1x1_d(x)
    branch_d = self.branch7x7_d(branch_d)

    out = [branch_a, branch_b, branch_c, branch_d]
    return torch.cat(out, 1)


class Quality_Estimator(nn.Module):
  """
  Quality Estimator based on VGGNet
  """
  def __init__(self):
    # dropout between fc layers?
    # batch normalization between conv layers?
    super(Quality_Estimator, self).__init__()
    self.features = None
    self.classifier = None

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


class IACNN(nn.Module):
  """
  Inception-based Artifact-Removal CNN
  """
  def __init__(self):
    # dimension
    super(IACNN, self).__init__()
    self.conv1 = BasicConv2d(1, 64, kernel_size=7)
    self.conv2 = BasicConv2d(64, 64, kernel_size=1)
    self.conv3 = BasicConv2d(64, 64, kernel_size=3)
    self.inception1 = Inception(64)
    self.inception2 = Inception(64)
    self.conv4 = BasicConv2d(64, 1, kernel_size=5)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.conv4(x)
    return x