import torch.nn as nn
from torch import Tensor
import torch
import inplace_abn
from copy import deepcopy as copy
from .bg_detector import BgDetector


class BaseNetwork(nn.Module):
    """Parent class used for network creation"""

    def __init__(self, n_channels=3, bilinear=True):
        """initialize instance

        Args:
            n_channels (int, optional): number of channels in input images. Defaults to 3.
            bilinear (bool, optional): upsampling used. Defaults to True. Defaults to False.
        """
        super(BaseNetwork, self).__init__()
        # hyperparameters sent to initialize the model used for cloning
        self.hyperparameters = {"n_channels": n_channels, "bilinear": bilinear}
        self.n_channels = n_channels
        self.bilinear = bilinear
        # backbone
        self.backbone = None
        # classification head
        self.classifier_head = None
        # output size of baseclf or backbone fed to classification head
        self.out_in_planes = None
        # whether to cache classification head output or not
        self.keep_clf_head_out = False
        # cached sem. logits
        self.latest_head_output = None
        # seen_not_seen detector
        self.seen_fg_network = None

    def clone(self):
        """Clones the current network into another copy

        Returns:
            BaseNetwork: cloned model
        """
        state_dict = self.state_dict()
        model = self.__class__(**self.hyperparameters)
        model.classifier_head = copy(self.classifier_head)
        model.seen_fg_network = copy(self.seen_fg_network)
        model.base_classifier = copy(self.base_classifier)
        model.backbone = copy(self.backbone)
        model.load_state_dict(state_dict)
        return model

    def enable_caching_sem_logits(self):
        """Enables saving classification head output"""
        self.keep_clf_head_out = True

    def pop_sem_logits(self):
        """pops cached sem logits

        Returns:
            Tensor: saved logits from the last forward pass
        """
        self.keep_clf_head_out = False
        sem_logits = self.latest_head_output
        self.latest_head_output = None
        return sem_logits

    def get_penultimate_output(self, x):
        """Override to return penultimate layre output for input image

        Args:
            x (tensor): input image
        """
        raise NotImplementedError("Override get_penultimate_output in your model!")

    def get_penultimate_layer_dim(self):
        """Override to return dimensions of the used penultimate layer"""
        raise NotImplementedError("Override get_penultimate_layer_dim in your model!")

    def get_seen_not_seen_head(self):
        """Returns the class used to initialize seen/not seen detector"""
        return BgDetector(self.get_penultimate_layer_dim())

    def get_classification_head(self, out_classes):
        """returns classification head

        Args:
            in_dim (int): input dim to clf head
            out_classes (int): number of output classes

        Returns:
            nn.Module: output clf head module
        """
        assert self.out_in_planes is not None
        return self._create_output_head(self.out_in_planes, out_classes)

    def _create_output_head(self, out_in_planes, n_classes):
        return nn.Conv2d(out_in_planes, n_classes, kernel_size=1)

    def _compute_classifier_output(self, x: Tensor):
        """compute output of classification head at current task

        Args:
            x (Tensor): input encoded feature

        Returns:
            tensor: prediction logits
        """
        if isinstance(self.classifier_head, nn.ModuleList):
            task_output = []
            for task_id in range(len(self.classifier_head)):
                task_output.append(self.classifier_head[task_id](x))
            x = torch.cat(task_output, dim=1)
        else:
            x = self.classifier_head(x)
        if self.keep_clf_head_out:
            # cache sem logits
            self.latest_head_output = x
        return x

    def get_parameters(self):
        """Returns network parameters or set of parameter groups

        Returns:
            list: list of dictionaries of parameter groups and specific information
        """
        return self.parameters()

    def forward(
        self,
        x: Tensor,
        return_attentions: bool = False,
        return_penultimate: bool = False,
        return_sem_logits: bool = False,
    ):
        """Forward pass over the network

        Args:
            x (Tensor): input image
            return_attentions (bool, optional): a flag to return a tuple of logits + feature attentions for PLOP. Defaults to False.
            return_penultimate (bool, optional): a flag to return a tuple of logits + penultimate outputs. Defaults to False.
            return_sem_logits (bool, optional): a flag to return a logits without upsampling. Defaults to False.

        Raises:
            NotImplementedError: Needs to be overriden
        """
        raise NotImplementedError("Should be overriden in children class!")

    def fix_bn(self):
        """Used to fix batch norm and ABN norms"""
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, inplace_abn.ABN)
                or isinstance(m, inplace_abn.InPlaceABN)
                or isinstance(m, inplace_abn.InPlaceABNSync)
            ):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
