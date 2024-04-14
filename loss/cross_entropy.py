from .base_loss import BaseLoss
import torch.nn.functional as F
import torch


class CrossEntropy(BaseLoss):
    """Basic CrossEntropy loss used for finetuning baseline"""

    def __init__(self, name="CrossEntropy", ignore_index=255):
        """initialize a new loss

        Args:
            name (str): name of the loss function
            ignore_index (int): index of the ignore pixel
        """
        super().__init__(name, ignore_index)

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            img (tensor): input image
            mask (tensor): target semantic segmentation map
            model (networks.BaseNetwork): model used for training
            train (bool): flag denoting this is train or val/test

        Returns:
            tuple: loss value, and output predictions
        """
        if isinstance(batch, dict):
            img = batch["main"][0]
            mask = batch["main"][1]
        else:
            img = batch[0]
            mask = batch[1]
        super(CrossEntropy, self).compute_loss(img, mask, model)
        loss, preds_mask =  self.compute_base_loss(img, mask, model, train=train)
        preds_output = preds_mask.argmax(dim=1)
        return loss, preds_output
