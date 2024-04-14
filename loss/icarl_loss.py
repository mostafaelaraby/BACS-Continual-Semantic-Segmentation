# Still in progress
from .base_loss import BaseLoss
from training.utils import freeze_network
from copy import deepcopy as copy
import torch
import torch.nn.functional as F
from training.loss_utils import IcarlCriterion

class IcarlLoss(BaseLoss):
    def __init__(self, name="Icarl", ignore_index=255):
        """Plop Loss initialization

        Args:
            name (str, optional): loss name. Defaults to "Icarl".
            ignore_index (int, optional): class index ignored during loss computation. Defaults to 255.
        """
        super().__init__(name, ignore_index)
        self.prev_model = None
        self.licarl = IcarlCriterion(reduction="mean", bkg=False)

    def on_train_start(self, task_num, **kwargs):
        """An event fired when task is switched in a CL setup 
        takes a set of named arguments
        """
        # move model to the right device 
        accelerator = kwargs.get("accelerator")
        if self.prev_model is not None:
            self.prev_model = self.prev_model.to(accelerator.root_device)

    def on_train_end(self, **kwargs):
        """An event fired at end of training to cache the network of previous task
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
            batch (tuple): a tuple of img and mask
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
        logits_new = model(img)
        entropy_loss = F.cross_entropy(logits_new, mask, ignore_index=self.ignore_index)
        if do_distillation:
            with torch.no_grad():
                logits_old = self.prev_model(img)
            loss = self.licarl(logits_new, mask, torch.sigmoid(logits_old))
            return (loss, logits_new.argmax(dim=1))
        return (
            entropy_loss,
            logits_new.argmax(dim=1),
        )

