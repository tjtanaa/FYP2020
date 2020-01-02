import torch
import torch.nn as nn


class Inception(nn.Module):
  """
  Inception module modified in "A Pseudo-Blind Convolutional Neural Network
  for the Reduction of Compression Artifacts"
  """
  def __init__(self):
    super(Inception, self).__init__()

  def forward(self, x):
    return x


class Quality_Estimator(nn.Module):
  """
  Quality Estimator based on VGGNet
  """
  def __init__(self):
    super(Quality_Estimator, self).__init__()
    self.features = None
    self.classifier = None

  def forward(self, x):
    return x


class IACNN(nn.Module):
  """
  Inception-based Artifact-Removal CNN
  """
  def __init__(self):
    super(IACNN, self).__init__()

  def forward(self, x):
    return x