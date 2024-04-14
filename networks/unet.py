""" Parts of the U-Net model From https://github.com/milesial/Pytorch-UNet/"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork
from torch import Tensor


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(BaseNetwork):
    def __init__(self, n_channels=3, bilinear=False, num_layers=5, **kwargs):
        """initialize a unet network

        Args:
            n_channels (int): number of input channels
            bilinear (bool, optional): flag for the upsampling. Defaults to False.
            num_layers (int, optional): number of layers for unet. Defaults to 5
        """
        super(UNet, self).__init__(n_channels=n_channels, bilinear=bilinear)
        self.num_layers = num_layers
        self.hyperparameters["num_layers"] = num_layers
        self._init_network(num_layers)

    def _init_network(self, num_layers=5):
        self.inc = DoubleConv(self.n_channels, 64)
        down_sampling_list = []
        feature_start = 64
        for _ in range(num_layers - 2):
            down_sampling_list.append(Down(feature_start, feature_start * 2))
            feature_start *= 2
        if self.bilinear:
            down_sampling_list.append(Down(feature_start, feature_start))
            feature_start *= 2
        else:
            down_sampling_list.append(Down(feature_start, feature_start * 2))
            feature_start *= 2
        self.downsample = nn.ModuleList(down_sampling_list)
        self.encoder_output_dim = feature_start // 2
        upsampling_list = []
        for _ in range(num_layers - 1):
            out_channels = feature_start // 2
            if self.bilinear:
                out_channels = feature_start // 4
            upsampling_list.append(Up(feature_start, out_channels, self.bilinear))
            feature_start //= 2
        self.upsample = nn.ModuleList(upsampling_list)
        self.out_in_planes = out_channels

    def get_penultimate_output(self, x):
        """Override to return penultimate layre output for input image

        Args:
            x (tensor): input image
        """
        x = self.inc(x)
        for down_sample_layer in self.downsample:
            x = down_sample_layer(x)
        return x

    def get_penultimate_layer_dim(self):
        """Override to return dimensions of the used penultimate layer"""
        return self.encoder_output_dim

    def forward(
        self,
        x: Tensor,
        return_attentions: bool = False,
        return_penultimate: bool = False,
    ):
        """Forward pass over UNET

        Args:
            x (Tensor): input image
            return_attentions (bool, optional): a flag to return a tuple of logits + feature attentions for PLOP. Defaults to False.
        Returns:
            tuple/tensor: Single prediction logits or a tuple of both predictions and attentions
        """

        def _add_attentions(x: Tensor):
            if return_attentions:
                attentions.append(x)

        attentions = []
        x_out = []
        x_out.append(self.inc(x))
        for layer_indx, down_sample_layer in enumerate(self.downsample):
            x_out.append(down_sample_layer(x_out[-1]))
            if layer_indx > 0:
                _add_attentions(x_out[-1])
        backbone_output = x_out
        for layer_indx, up_sample_layer in enumerate(self.upsample):
            x_out[-1] = up_sample_layer(x_out[-1], x_out[-2 - layer_indx])
        logits = self._compute_classifier_output(x_out[-1])
        if return_penultimate and return_attentions:
            return logits, backbone_output, attentions
        elif return_penultimate:
            return logits, backbone_output
        elif return_attentions:
            return logits, attentions
        return logits


if __name__ == "__main__":
    model = UNet(3, bilinear=False)
    model.classifier_head = model.get_classification_head(2)
    input_image = torch.rand((5, 3, 224, 224))
    output_image = model(input_image)
    assert output_image.shape == torch.Size([5, 2, 224, 224])
