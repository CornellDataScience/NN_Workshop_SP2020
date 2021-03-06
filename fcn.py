import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.segmentation.fcn import FCNHead


class ConvBlock(nn.Module):
  """
  On an input tensor x, performs: [Conv3x3, BatchNorm, ReLU] x 2 \n
  For an input tensor x of shape (batch_size, #channels, height, width) \n
  Requires: \n
    `in_channels`: number of channels (or depth) of input tensor \n
    `out_channels`: number of channnels (or depth) of output tensor \n
  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    ## ===========================================================
    ## BEGIN: YOUR CODE.
    ## ===========================================================

    ## ===========================================================
    ## END: YOUR CODE.
    ## ===========================================================
  
  def forward(self, x):
    ## ===========================================================
    ## BEGIN: YOUR CODE.
    ## ===========================================================

    ## ===========================================================
    ## END: YOUR CODE.
    ## ===========================================================
    return x


class DownSample(nn.Module):
  """
  On an input tensor x, downsamples its height and width dimensions by x0.5
  after using a `ConvBlock`. \n
  For an input tensor x of shape (batch_size, #channels, height, width) \n
  Requires:\n
    `in_channels`: number of channels (or depth) of input tensor \n
    `out_channels`: number of channnels (or depth) of output tensor \n
  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    ## ===========================================================
    ## BEGIN: YOUR CODE.
    ## ===========================================================

    ## ===========================================================
    ## END: YOUR CODE.
    ## ===========================================================
  
  def forward(self, x):
    ## ===========================================================
    ## BEGIN: YOUR CODE.
    ## ===========================================================

    ## ===========================================================
    ## END: YOUR CODE.
    ## ===========================================================
    return x


class UpSample(nn.Module):
  """
  On an input tensor x, upsamples its height and width dimensions by x2,
  and follows the upsampling by applying a `ConvBlock`. \n
  For an input tensor x of shape (batch_size, #channels, height, width) \n
  Requires: \n
    `in_channels`: number of channels (or depth) of input tensor \n
    `out_channels`: number of channnels (or depth) of output tensor \n
  """
  def __init__(self, in_channels, out_channels, use_deconv=True):
    super().__init__()
    self.use_deconv = use_deconv

    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_block = ConvBlock(out_channels, out_channels)
  
  def forward(self, x):
    x = self.up(x)
    x = self.conv_block(x)
    return x


class FCN(nn.Module):
  def __init__(self, num_classes=1):
    super(FCN, self).__init__()

    # First Conv block takes in image of dimension (batch_size, 3, h, w)
    # 3 channels for RGB
    self.block1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
    )

    ## ===========================================================
    ## BEGIN: YOUR CODE.
    ## ===========================================================

    ## ===========================================================
    ## END: YOUR CODE.
    ## ===========================================================

    self.depth_reduce = ConvBlock(256, 64)
    self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    

  def forward(self, x):
    input_shape = x.shape[-2:]   # (h, w), #c = 3

    ## ===========================================================
    ## BEGIN: YOUR CODE.
    ## ===========================================================
    x_out = None

    ## ===========================================================
    ## END: YOUR CODE.
    ## ===========================================================
    x_out = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=False)
    x_out = self.final_conv(x_out)
    return x_out


def load_fcn(num_classes=1, from_checkpoint=None):
  """
  Creates an FCN model, and if specified, load's weights from a
  check point file.\n
  Requires:\n
    `num_classes`: Number of classes that the model should expect to output.\n
    `from_checkpoint`: (string) path to model checkpoint file containing weights\n
  """
  model = FCN(num_classes=num_classes)

  if from_checkpoint:
    assert os.path.isfile(from_checkpoint),\
      "Model's .pth or .bin checkpoint file does not exist"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    state_dict = torch.load(from_checkpoint, map_location=device)
    model.load_state_dict(state_dict)

  return model

  
# Small test
if __name__ == "__main__":
  fcn = load_fcn(num_classes=1)
  x = torch.randn(1, 3, 224, 224)
  out = fcn(x)
  print(out.shape)
  assert out.shape == (1, 1, 224, 224)