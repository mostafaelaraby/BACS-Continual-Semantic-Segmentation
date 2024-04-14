from collections import OrderedDict
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as functional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import os
import copy


class ResidualBlock(nn.Module):
    """Configurable residual block

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of channels in the internal feature maps. Can either have two or three elements: if three construct
        a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
        `3 x 3` then `1 x 1` convolutions.
    stride : int
        Stride of the first `3 x 3` convolution
    dilation : int
        Dilation to apply to the `3 x 3` convolutions.
    groups : int
        Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
        bottleneck blocks.
    norm_act : callable
        Function to create normalization / activation Module.
    dropout: callable
        Function to create Dropout Module.
    """

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dilation=1,
        groups=1,
        norm_act=nn.BatchNorm2d,
        dropout=None,
        last=False,
    ):
        super(ResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        if not is_bottleneck:
            bn2 = norm_act(channels[1])
            bn2.activation = "identity"
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        channels[0],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                ),
                ("bn1", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                ),
                ("bn2", bn2),
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            bn3 = norm_act(channels[2])
            bn3.activation = "identity"
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels, channels[0], 1, stride=1, padding=0, bias=False
                    ),
                ),
                ("bn1", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        groups=groups,
                        dilation=dilation,
                    ),
                ),
                ("bn2", norm_act(channels[1])),
                (
                    "conv3",
                    nn.Conv2d(
                        channels[1], channels[2], 1, stride=1, padding=0, bias=False
                    ),
                ),
                ("bn3", bn3),
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False
            )
            self.proj_bn = norm_act(channels[-1])
            self.proj_bn.activation = "identity"

        self._last = last

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual)
        else:
            residual = x
        x = self.convs(x) + residual

        if self.convs.bn1.activation == "leaky_relu":
            act = functional.leaky_relu(
                x,
                negative_slope=self.convs.bn1.activation_param,
                inplace=not self._last,
            )
        elif self.convs.bn1.activation == "relu":
            act = functional.relu(
                x,
                inplace=not self._last,
            )
        elif self.convs.bn1.activation == "elu":
            act = functional.elu(
                x, alpha=self.convs.bn1.activation_param, inplace=not self._last
            )
        elif self.convs.bn1.activation == "identity":
            act = x

        if self._last:
            return act, x
        return act


class IdentityResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dilation=1,
        groups=1,
        norm_act=nn.BatchNorm2d,
        dropout=None,
    ):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        channels[0],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                ),
                ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                ),
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        channels[0],
                        1,
                        stride=stride,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        groups=groups,
                        dilation=dilation,
                    ),
                ),
                ("bn3", norm_act(channels[1])),
                (
                    "conv3",
                    nn.Conv2d(
                        channels[1], channels[2], 1, stride=1, padding=0, bias=False
                    ),
                ),
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False
            )

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out


class ResNet(nn.Module):
    """Standard residual network

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the four modules of the network
    bottleneck : bool
        If `True` use "bottleneck" residual blocks with 3 convolutions, otherwise use standard blocks
    norm_act : callable
        Function to create normalization / activation Module
    classes : int
        If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
        of the network
    dilation : int or list of int
         List of dilation factors for the four modules of the network, or `1` to ignore dilation
    keep_outputs : bool
        If `True` output a list with the outputs of all modules
    """

    def __init__(
        self,
        structure,
        bottleneck,
        norm_act=nn.BatchNorm2d,
        output_stride=16,
    ):
        super(ResNet, self).__init__()
        self.structure = structure
        self.bottleneck = bottleneck

        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if output_stride != 8 and output_stride != 16:
            raise ValueError("Output stride must be 8 or 16")

        if output_stride == 16:
            dilation = [1, 1, 1, 2]  # dilated conv for last 3 blocks (9 layers)
        elif output_stride == 8:
            dilation = [1, 1, 2, 4]  # 23+3 blocks (78 layers)
        else:
            raise NotImplementedError

        self.dilation = dilation

        # Initial layers
        layers = [
            ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
            ("bn1", norm_act(64)),
            ("pool1", nn.MaxPool2d(3, stride=2, padding=1)),
        ]
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 256)
        else:
            channels = (64, 64)
        self.aux_channels_out = 0
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                stride, dil = self._stride_dilation(dilation, mod_id, block_id)
                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        ResidualBlock(
                            in_channels,
                            channels,
                            norm_act=norm_act,
                            stride=stride,
                            dilation=dil,
                            last=block_id == num - 1,
                        ),
                    )
                )

                # Update channels and p_keep
                in_channels = channels[-1]
            if mod_id == 2:
                self.aux_channels_out = in_channels
            # Create module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

            # Double the number of channels for the next module
            channels = [c * 2 for c in channels]

        self.out_channels = in_channels

    @staticmethod
    def _stride_dilation(dilation, mod_id, block_id):
        d = dilation[mod_id]
        s = 2 if d == 1 and block_id == 0 and mod_id > 0 else 1
        return s, d

    def forward(self, x):
        attentions = []

        x = self.mod1(x)
        # attentions.append(x)

        x, att = self.mod2(x)
        attentions.append(att)

        x, att = self.mod3(x)
        attentions.append(att)

        x, att = self.mod4(x)
        attentions.append(att)

        x, att = self.mod5(x)
        attentions.append(att)

        return x, attentions


_NETS = {
    "resnet18": {"structure": [2, 2, 2, 2], "bottleneck": False},
    "resnet34": {"structure": [3, 4, 6, 3], "bottleneck": False},
    "resnet50": {"structure": [3, 4, 6, 3], "bottleneck": True},
    "resnet101": {"structure": [3, 4, 23, 3], "bottleneck": True},
    "resnet152": {"structure": [3, 8, 36, 3], "bottleneck": True},
}

# Pretrained on ImageNet
model_urls = {
    "resnet101_abn": "https://github.com/arthurdouillard/CVPR2021_PLOP/releases/download/v1.0/resnet101_iabn_sync.pth.tar",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def create_resnet(arch, norm_act, output_stride=16, pretrained=False, **kwargs):
    assert arch in _NETS
    model = ResNet(
        _NETS[arch]["structure"],
        +_NETS[arch]["bottleneck"],
        norm_act=norm_act,
        output_stride=output_stride,
    )
    is_abn = "abn" in norm_act.func.__module__
    if pretrained:
        weights_path = kwargs.get("resnet_weights_disk", None)
        if weights_path is None:
            abn_key = "{}_abn".format(arch) if is_abn else arch
            if (
                norm_act is not None
                and hasattr(norm_act, "func")
                and abn_key in model_urls
            ):
                
                pretrained_dict = load_state_dict_from_url(
                    model_urls[abn_key], progress=True, map_location="cpu"
                )
                if  is_abn:
                    pretrained_dict = pretrained_dict["state_dict"]
            else:
                raise Exception(
                    "Pretraining not supported when using BatchNorm activation use ABN instead"
                )
        else:
            assert os.path.isfile(weights_path)
            pretrained_dict = torch.load(weights_path, map_location="cpu")["state_dict"]
        if is_abn:
            for key in copy.deepcopy(list(pretrained_dict.keys())):
                pretrained_dict[key[7:]] = pretrained_dict.pop(key)
            del pretrained_dict["classifier.fc.weight"]
            del pretrained_dict["classifier.fc.bias"]
            state_dict = pretrained_dict
        else:
            state_dict = model.state_dict()
            for old_key, new_key in zip(list(state_dict.keys()),list(pretrained_dict.keys())):
                if state_dict[old_key].shape == pretrained_dict[new_key].shape:
                    state_dict[old_key] = pretrained_dict[new_key]
        model.load_state_dict(state_dict)
    return model
