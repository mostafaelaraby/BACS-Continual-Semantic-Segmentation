from .baselearner import BaseLearner
from networks.base_network import BaseNetwork
import torch
from torch import nn
from visualization import Mode
from functools import partial


class MultiHeadLearner(BaseLearner):
    def __init__(self, network: BaseNetwork, config, is_domain_shift):
        super().__init__(network, config, is_domain_shift)

    def _initialize_head(self, bg_head, current_head, new_classes):
        """Creates classification head of current task

        Args:
            bg_head (nn.Module): Head used to predict the background
            current_head (nn.Module): new head created
            new_classes (int): number of classes in the new task

        Returns:
            nn.Module: current head pytorch module
        """
        def _get_head(head):
            if isinstance(bg_head, nn.Sequential):
                return head[-1]
            return head  
        imprinting_w = _get_head(bg_head).weight[0]
        bkg_bias = _get_head(bg_head).bias[0] 
        bias_diff = torch.log(
            torch.FloatTensor([new_classes + 1], device=bkg_bias.device)
        )
        new_bias = bkg_bias - bias_diff
        _get_head(current_head).weight.data.copy_(imprinting_w)
        _get_head(current_head).bias.data.copy_(new_bias)
        _get_head(bg_head).bias[0].data.copy_(new_bias.squeeze(0))

    def on_init_start(self, trainer):
        """Called on trainer init before init of network

        Args:
            trainer (pl.Trainer): Pytorch lightning trainer
        """
        # initialize networks' classifier
        if self.task_id == 0:
            num_classes =  self.config.dataset.dataset.num_classes if self.is_domain_shift else self.config.training.initial_increment + 1
            self.network.classifier_head = nn.ModuleList(
                [self.network.get_classification_head(num_classes)]
            )
        else:
            num_classes = self.config.training.increment
            current_head = self.network.get_classification_head(num_classes)
            self._initialize_head(
                self.network.classifier_head[0],
                current_head,
                num_classes,
            )
            self.network.classifier_head.append(current_head)
        if self.use_bg_detector and not (self.is_domain_shift):
            self._init_bg_detector()

    def teardown(self, trainer, pl_module, stage=None):
        """Called on training end

        Args:
            trainer (pl.trainer): Pytorch lightning trainer
            pl_module (pl.lightningmodule): lightning module including our network used in training
        """
        # update loss function
        if stage == "fit":
            # create a new data loader with false shuffle
            pl_module.loss_fn.on_train_end(
                trainer=trainer, 
                pl_module=pl_module,
                model=pl_module.network,
                train_dataloader=trainer.datamodule.train_dataloader(),
                accelerator=trainer.accelerator,
                log_media=partial(pl_module.log_media.append, mode=Mode.TRAIN)
                if pl_module.log_images
                else None,
                pre_last_tasks=self.task_id < trainer.datamodule.n_tasks - 1,
            )
            self.task_id += 1
