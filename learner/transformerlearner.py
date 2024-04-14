from .multiheadlearner import MultiHeadLearner
from networks.base_network import BaseNetwork
from networks.utils import trunc_normal_
import torch
from torch import nn


class TransformerLearner(MultiHeadLearner):
    def __init__(self, network: BaseNetwork, config, is_domain_shift):
        super().__init__(network, config, is_domain_shift)
        self.new_token_init = config.training.new_token_init

    def on_init_start(self, trainer):
        """Called on trainer init before init of network

        Args:
            trainer (pl.Trainer): Pytorch lightning trainer
        """
        if self.use_bg_detector and not (self.is_domain_shift):
            self._init_bg_detector()
        # initialize networks' classifier
        device = self.network.base_classifier.class_tokens.device
        if self.task_id == 0 and self.is_domain_shift:
            self.network.base_classifier.class_tokens = nn.Parameter(
                torch.randn(
                    1,
                    self.config.dataset.dataset.num_classes,
                    self.network.transformer_config.hidden_dim,
                )
            ).to(device)
            trunc_normal_(self.network.base_classifier.class_tokens, std=0.02)
            self.network.base_classifier.mask_norm = nn.LayerNorm(
                self.config.dataset.dataset.num_classes
            )
        elif self.task_id == 0:
            self.network.base_classifier.class_tokens = nn.Parameter(
                torch.randn(
                    1,
                    self.config.training.initial_increment + 1,
                    self.network.transformer_config.hidden_dim,
                )
            ).to(device)
            trunc_normal_(self.network.base_classifier.class_tokens, std=0.02)
            self.network.base_classifier.mask_norm = nn.LayerNorm(
                self.config.training.initial_increment + 1
            )
        elif not(self.is_domain_shift):
            if self.new_token_init == "random":
                # Initialize new class tokens and mask layer norm using randomly sampled values
                current_class_tokens = torch.randn(
                    1,
                    self.config.training.increment,
                    self.network.transformer_config.hidden_dim,
                )
                trunc_normal_(current_class_tokens, std=0.02)

                current_mask_norm = self.network.base_classifier.mask_norm
                current_normalized_shape = current_mask_norm.normalized_shape[0]

                new_mask_norm = nn.LayerNorm(
                    current_normalized_shape + self.config.training.increment
                )

                with torch.no_grad():
                    new_mask_norm.weight[
                        0:current_normalized_shape
                    ] = current_mask_norm.weight
                    new_mask_norm.bias[
                        0:current_normalized_shape
                    ] = current_mask_norm.bias

            elif self.new_token_init == "background":
                # Initialize new class tokens and mask layer norm using values from the background class
                current_class_tokens = (
                    self.network.base_classifier.class_tokens[:, 0:1, :]
                    .detach()
                    .repeat(1, self.config.training.increment, 1)
                )

                current_mask_norm = self.network.base_classifier.mask_norm
                current_normalized_shape = current_mask_norm.normalized_shape[0]

                new_mask_norm = nn.LayerNorm(
                    current_normalized_shape + self.config.training.increment
                )
                with torch.no_grad():
                    new_mask_norm.weight[
                        0:current_normalized_shape
                    ] = current_mask_norm.weight
                    new_mask_norm.bias[
                        0:current_normalized_shape
                    ] = current_mask_norm.bias

                    new_mask_norm.weight[
                        -self.config.training.increment :
                    ] = current_mask_norm.weight[0:1].repeat(
                        self.config.training.increment
                    )
                    new_mask_norm.bias[
                        -self.config.training.increment :
                    ] = current_mask_norm.bias[0:1].repeat(
                        self.config.training.increment
                    )

            elif self.new_token_init == "mean":
                # Initialize new class tokens and mask layer norm using values using mean of all tokens
                current_class_tokens = (
                    self.network.base_classifier.class_tokens.mean(1, keepdims=True)
                    .detach()
                    .repeat(1, self.config.training.increment, 1)
                )

                current_mask_norm = self.network.base_classifier.mask_norm
                current_normalized_shape = current_mask_norm.normalized_shape[0]

                new_mask_norm = nn.LayerNorm(
                    current_normalized_shape + self.config.training.increment
                )
                with torch.no_grad():
                    new_mask_norm.weight[
                        0:current_normalized_shape
                    ] = current_mask_norm.weight
                    new_mask_norm.bias[
                        0:current_normalized_shape
                    ] = current_mask_norm.bias
                    new_mask_norm.weight[
                        -self.config.training.increment :
                    ] = current_mask_norm.weight.mean().repeat(
                        self.config.training.increment
                    )
                    new_mask_norm.bias[
                        -self.config.training.increment :
                    ] = current_mask_norm.weight.mean().repeat(
                        self.config.training.increment
                    )

            # Init while severing link to parameters
            self.network.base_classifier.class_tokens = nn.Parameter(
                torch.cat(
                    (self.network.base_classifier.class_tokens, current_class_tokens),
                    dim=1,
                ).to(device)
            )
            self.network.base_classifier.mask_norm = new_mask_norm
