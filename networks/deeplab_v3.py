# Implementation based on Torch Vision
import os
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from .base_network import BaseNetwork
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
from functools import partial
from .resnet import create_resnet
from .resnet_atrous import create_resnet_atrous
from .utils import SynchronizedBatchNorm2d, InPlaceABRSync, InPlaceABR, ABR
from pytorch_lightning.utilities import rank_zero_warn

models_urls = {
    "deeplabv3_resnet50_coco": "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_resnet101_coco": "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
}


class DeepLabV3(BaseNetwork):
    def __init__(self, **kwargs) -> None:
        super(DeepLabV3, self).__init__(
            n_channels=kwargs.get("n_channels", 3),
            bilinear=kwargs.get("bilinear", True),
        )
        self.hyperparameters = kwargs
        self.pretrained_backbone = kwargs.get("pretrained_backbone", False)
        self.output_stride = kwargs.get("output_stride", 16)
        self.use_atrous_encoder = kwargs.get("atrous_encoder", False) 
        self.use_separate_feats = kwargs.get("separate_feats", False) 
        # path on disk for backbone weights can be None and in that case
        # url will be used
        resnet_backbone_path = (
            os.path.expanduser(kwargs.get("backbone_weights_path", None))
            if kwargs.get("backbone_weights_path", None) is not None
            else None
        )
        if resnet_backbone_path is not None and not (
            os.path.isfile(resnet_backbone_path)
        ):
            rank_zero_warn(
                "Resnet Backbone file path doesn't exist {}".format(
                    resnet_backbone_path
                )
            )
            resnet_backbone_path = None
        self.norm = kwargs.get("norm", "bn")
        self.norm_act = None
        self._init_norm_act()
        if self.use_atrous_encoder:
            self.backbone = create_resnet_atrous(
                "resnet101",
                output_stride=self.output_stride,
                pretrained=self.pretrained_backbone,
                resnet_weights_disk=resnet_backbone_path,
            )
        else:
            self.backbone = create_resnet(
                "resnet101",
                self.norm_act,
                pretrained=self.pretrained_backbone,
                resnet_weights_disk=resnet_backbone_path,
            )
        self.out_in_planes = 256
        self.base_classifier = DeepLabHead(
            self.backbone.out_channels, self.out_in_planes, 256, norm_act=self.norm_act
        )
        self.classifier_head = None

    def _init_norm_act(self):
        # init norm activation
        if self.norm == "bn" or self.use_atrous_encoder:
            self.norm_act = partial(
                BNReLUAct, momentum=0.0003, use_sync_batch=self.use_atrous_encoder
            )
        elif self.norm == "iabn_sync":
            self.norm_act = partial(
                InPlaceABNSync, activation="leaky_relu", activation_param=0.01
            )
        elif self.norm == "iabn":
            self.norm_act = partial(
                InPlaceABN, activation="leaky_relu", activation_param=0.01
            )
        elif self.norm == "abn":
            self.norm_act = partial(ABN, activation="leaky_relu", activation_param=0.01)
        elif self.norm == "iabr_sync":
            self.norm_act = partial(
                InPlaceABRSync, activation="leaky_relu", activation_param=0.01, momentum=0.
            )
        elif self.norm == "iabr":
            self.norm_act = partial(
                InPlaceABR, activation="leaky_relu", activation_param=0.01, momentum=0.
            )
        elif self.norm == "abr":
            self.norm_act = partial(ABR, activation="leaky_relu", activation_param=0.01, momentum=0.)
        else:
            raise NotImplementedError(
                "Selected Norm {} is not supported ".format(self.norm)
            )

    def _create_output_head(self, out_in_planes, n_classes):
        if self.use_separate_feats:
            return nn.Sequential(
                                nn.Conv2d(out_in_planes, out_in_planes, 3, padding=1, bias=False),
                                self.norm_act(out_in_planes),
                                nn.Conv2d(out_in_planes, n_classes, 1))
        return nn.Conv2d(out_in_planes, n_classes, kernel_size=1)

    def get_penultimate_output(self, x):
        """Override to return penultimate layre output for input image

        Args:
            x (tensor): input image
        """
        backbone_out, _ = self.backbone(x)
        if hasattr(self, "seen_fg_network") and self.seen_fg_network is not None:
            backbone_out = self.seen_fg_network.get_penultimate_output(backbone_out)
        return backbone_out

    def get_penultimate_layer_dim(self):
        """Override to return dimensions of the used penultimate layer"""
        if hasattr(self, "seen_fg_network") and self.seen_fg_network is not None:
            return self.seen_fg_network.get_penultimate_layer_dim()
        return self.backbone.out_channels

    def forward(
        self,
        x: Tensor,
        return_attentions: bool = False,
        return_penultimate: bool = False,
        return_sem_logits: bool = False,
        only_attentions: bool = False,
    ):
        """Forward pass over the network

        Args:
            x (Tensor): input image
            return_attentions (bool, optional): a flag to return a tuple of logits + feature attentions for PLOP. Defaults to False.
            return_penultimate (bool, optional): a flag to return a tuple of logits + penultimate outputs. Defaults to False.
            return_sem_logits (bool, optional): a flag to return a logits without upsampling. Defaults to False.

        Returns:
            tuple/tensor: Single prediction logits or a tuple of both predictions and attentions
        """
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        backbone_output, attentions = self.backbone(x)
        x = self.base_classifier(backbone_output)
        # POD loss operate on output from backbone layers and from DeepLabHead output
        attentions.append(x)
        if only_attentions:
            return attentions
        sem_logits = self._compute_classifier_output(x)
        if return_sem_logits:
            logits = sem_logits
        else:
            logits = F.interpolate(
                sem_logits, size=input_shape, mode="bilinear", align_corners=False
            ) 
        if return_penultimate:
            if hasattr(self, "seen_fg_network") and self.seen_fg_network is not None:
                backbone_output = self.seen_fg_network.get_penultimate_output(
                    backbone_output
                )
        if return_penultimate and return_attentions:
            return logits, backbone_output, attentions
        elif return_penultimate:
            return logits, backbone_output
        elif return_attentions:
            return logits, attentions
        return logits


class DeepLabHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=256,
        out_stride=16,
        norm_act=nn.BatchNorm2d,
        pooling_size=None,
    ):
        super(DeepLabHead, self).__init__()
        self.pooling_size = pooling_size

        if out_stride == 16:
            dilations = [6, 12, 18]
        elif out_stride == 8:
            dilations = [12, 24, 32]

        self.map_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilations[0],
                    padding=dilations[0],
                ),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilations[1],
                    padding=dilations[1],
                ),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilations[2],
                    padding=dilations[2],
                ),
            ]
        )
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(
            in_channels, hidden_channels, 1, bias=False
        )
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.map_bn, "activation"):
            gain = nn.init.calculate_gain(
                self.map_bn.activation, self.map_bn.activation_param
            )
        else:
            gain = nn.init.calculate_gain("linear")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(
            x
        )  # if training is global avg pooling 1x1, else use larger pool size
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        return out

    def _get_pooling_size(self, x):
        def try_index(pooling_size, index):
            if isinstance(pooling_size, list):
                return pooling_size[index]
            return pooling_size

        pooling_size = (
            min(try_index(self.pooling_size, 0), x.shape[2]),
            min(try_index(self.pooling_size, 1), x.shape[3]),
        )
        return pooling_size

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            # this is like Adaptive Average Pooling (1,1)
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = self._get_pooling_size(x)
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2
                if pooling_size[1] % 2 == 1
                else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2
                if pooling_size[0] % 2 == 1
                else (pooling_size[0] - 1) // 2 + 1,
            )

            pool = F.avg_pool2d(x, pooling_size, stride=1)
            pool = F.pad(pool, pad=padding, mode="replicate")
        return pool


class BNReLUAct(nn.Module):
    def __init__(
        self, num_features, eps=1e-05, momentum=0.1, affine=True, use_sync_batch=False
    ):
        super(BNReLUAct, self).__init__()
        self.activation = "relu"
        self.activation_param = 0
        if use_sync_batch:
            self.bn = SynchronizedBatchNorm2d(
                num_features=num_features, eps=eps, momentum=momentum, affine=affine
            )
        else:
            self.bn = nn.BatchNorm2d(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
            )
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        if self.activation != "identity":
            x = self.act_layer(x)
        return x


if __name__ == "__main__":
    # Testing the implementation
    model = DeepLabV3()
    model.classifier_head = model.get_classification_head(2)
    input_image = torch.rand((5, 3, 224, 224))
    output_image = model(input_image)
    assert output_image.shape == torch.Size([5, 2, 224, 224])
