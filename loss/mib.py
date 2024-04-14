# Still in progress
from .base_loss import BaseLoss
from training.utils import freeze_network
from copy import deepcopy as copy
import torch
from torch import nn
from training.loss_utils import UnbiasedCrossEntropy, UnbiasedKnowledgeDistillationLoss


class MiB(BaseLoss):
    def __init__(self, name="MiB", ignore_index=255, bg_weighted_ce: bool = False,):
        """Plop Loss initialization

        Args:
            name (str, optional): loss name. Defaults to "Icarl".
            ignore_index (int, optional): class index ignored during loss computation. Defaults to 255.
        """
        super().__init__(name, ignore_index)
        self.prev_model = None
        self.ubiased_ce = None  # will need access to old classes
        self.bg_weighted_ce = bg_weighted_ce
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=1.0)
        self.lkd = 10

    def on_train_start(self, task_num, **kwargs):
        """An event fired when task is switched in a CL setup 
        takes a set of named arguments
        """
        self.ubiased_ce = UnbiasedCrossEntropy(
            old_cl=self.old_classes, ignore_index=self.ignore_index, reduction="none"
        )
        # move model to the right device 
        accelerator = kwargs.get("accelerator")
        if self.prev_model is not None:
            self.prev_model = self.prev_model.to(accelerator.root_device)

    def on_train_end(self, **kwargs):
        """An event fired at end of testing to cache the network of previous task
        """     
        super().on_train_end(**kwargs)
        pre_last_tasks = kwargs.get("pre_last_tasks")
        if not(pre_last_tasks):
            return
        model = kwargs.get("model") if "model" in kwargs else None
        if model is not None:
            self.prev_model = model.clone()
            freeze_network(self.prev_model) 

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            batch (tuple): tuple of img tensor and mask
            model (networks.BaseNetwork): model used for training
            train (bool): flag denoting this is train or val/test
            
        Raises:
            NotImplementedError: needs to be overriden in child classes
        """
        if isinstance(batch, dict):
            img = batch["main"][0]
            mask = batch["main"][1]
        else:
            img = batch[0]
            mask = batch[1]
        # Feature distillation from previous task network if we have it
        do_distillation = self.prev_model is not None and train
        if not(self.bg_weighted_ce) or not(train): 
            logits_new = model(img)
        if self.old_classes != 0 and train and not(self.bg_weighted_ce): 
            loss = self.ubiased_ce(logits_new, mask).mean()
        elif  self.bg_weighted_ce and train:
            # Our loss instead of pseudo labelling
            loss, logits_new = self.compute_base_loss(
                img, mask, model, train=train, use_weighted_ce=(self.old_classes != 0)
            )
        else:
            loss = self.ce(logits_new, mask).mean()
        if do_distillation:
            with torch.no_grad():
                logits_old = self.prev_model(img)
            loss += self.lkd * self.lkd_loss(logits_new, logits_old)
            return (loss, logits_new.argmax(dim=1))
        return (
            loss,
            logits_new.argmax(dim=1),
        )

