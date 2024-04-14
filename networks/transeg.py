# Implementation based on Torch Vision
import collections
import torch
import einops
import math
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .deeplab_v3 import DeepLabV3
from .layers import Block
from .utils import init_weights, trunc_normal_


class TranSeg(DeepLabV3):
    def __init__(self, **kwargs) -> None:
        super(TranSeg, self).__init__(
            **kwargs,
        )

        self.crop_size = kwargs.get("crop_size")
        self.num_classes = kwargs.get("num_classes")
        self.transformer_config = kwargs.get("transformer")

        self.base_classifier = TransformerHead(
            self.backbone.out_channels,
            self.crop_size,
            self.num_classes,
            self.transformer_config,
        )

    def forward(
        self,
        x: Tensor,
        return_attentions: bool = False,
        return_penultimate: bool = False,
        return_sem_logits: bool = False,
        only_attentions: bool = False
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
        if only_attentions:
            return self.base_classifier(backbone_output, only_attentions=True)
        x, trans_atts = self.base_classifier(backbone_output, return_attentions=True)
        attentions += trans_atts
        if self.keep_clf_head_out:
            # cache sem logits
            self.latest_head_output = x
        if not (return_sem_logits):
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        logits = x

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


#########################
### Transformer Model ###
#########################


class TransformerHead(nn.Module):
    def __init__(self, in_channels, crop_size, num_classes=None, config=None):
        super(TransformerHead, self).__init__()

        self.d_model = config.hidden_dim
        self.scale = self.d_model ** -0.5

        self.dim_feedforward = config.dim_feedforward
        self.feature_embedding = torch.nn.Conv2d(
            in_channels, self.d_model, kernel_size=(1, 1)
        )

        # We assume a shrinkage size of 16 times. #ToDo Find a better way to determine number of patches.
        if isinstance(crop_size, collections.Iterable):
            self.num_patches = [int(crop_size[0] / 16), int(crop_size[1] / 16)]
        else:
            self.num_patches = [int(crop_size / 16) for _ in range(2)]
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.d_model, self.num_patches[0], self.num_patches[1])
        )

        # Final number of classes only needed for joint setting.
        if num_classes is not None:
            self.class_tokens = nn.Parameter(torch.randn(1, num_classes, self.d_model))
            trunc_normal_(self.class_tokens, std=0.02)
            self.mask_norm = nn.LayerNorm(num_classes)
        else:
            # Initialize self.class_tokens to be replaced later.
            self.class_tokens = None
            self.mask_norm = None

        # Decoder Architecture.
        self.blocks = nn.ModuleList(
            [
                Block(self.d_model, config.nhead, self.dim_feedforward, dropout=0.0)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.proj_patch = nn.Parameter(
            self.scale * torch.randn(self.d_model, self.d_model)
        )
        self.proj_classes = nn.Parameter(
            self.scale * torch.randn(self.d_model, self.d_model)
        )
        self.decoder_norm = nn.LayerNorm(self.d_model)

        # Initialize Weights
        self.apply(init_weights)

    def forward(self, x, return_attentions=False, only_attentions=False):
        # Get input shape to be used for reshaping the target below
        height, width = x.shape[2:]
        attentions = []
        # Extract features using ResNet. Add positional embeddings.
        x = self.feature_embedding(
            x
        )  # batch_size x d_model x w x h. Channels go from 2048 --> out_channels
        x = x + self.pos_embed  # batch_size x out_channels x w x h.
        x = einops.rearrange(x, "b c h w -> b (h w) c")  # batch_size x (h*w) x d_model

        # Append class tokens to pixel tensor
        class_tokens = self.class_tokens.expand(x.size(0), -1, -1)
        n_cls = class_tokens.shape[1]
        x = torch.cat((x, class_tokens), 1)

        # Apply decoder layers
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # Split patch tokens and class tokens.
        patches, cls_seg_feat = x[:, :-n_cls], x[:, -n_cls:]
        # re-order patches
        image_feats = einops.rearrange(
            patches, "b (h w) n -> b n h w", h=height
        ).contiguous()
        attentions.append(image_feats)
        if only_attentions:
            return attentions
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        # Perform dot product to recover final output masks.
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = einops.rearrange(masks, "b (h w) n -> b n h w", h=height).contiguous()
        if return_attentions:
            return masks, attentions
        return masks


if __name__ == "__main__":
    # Testing the implementation
    from omegaconf import DictConfig

    model = TranSeg(
        crop_size=224,
        num_classes=2,
        transformer=DictConfig(
            {
                "hidden_dim": 256,
                "nhead": 2,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
            }
        ),
    )
    model.classifier_head = model.get_classification_head(2)
    input_image = torch.rand((5, 3, 224, 224))
    output_image = model(input_image)
    assert output_image.shape == torch.Size([5, 2, 224, 224])
