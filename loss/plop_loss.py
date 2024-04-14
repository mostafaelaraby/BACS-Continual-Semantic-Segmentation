# Still in progress
from .base_loss import BaseLoss
from training.loss_utils import features_distillation, entropy
from training.utils import freeze_network
from copy import deepcopy as copy
import torch
from training.utils import find_median
from torch import nn


class PlopLoss(BaseLoss):
    def __init__(self, name="Plop", ignore_index=255, bg_weighted_ce: bool = False):
        """Plop Loss initialization

        Args:
            name (str, optional): loss name. Defaults to "Plop".
            ignore_index (int, optional): class index ignored during loss computation. Defaults to 255.
        """
        super().__init__(name, ignore_index)
        self.prev_model = None
        self.pseudo_ablation = None  # can be corrected_errors, removed_errors
        self.thresholds = None
        self.max_entropy = None
        self.use_logits = (
            True  # to add logits to the attention in feature distillation loss
        )
        self.classif_adaptive_factor = True
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, reduction="none"
        )
        self.bg_weighted_ce = bg_weighted_ce

    def on_train_start(self, task_num, **kwargs):
        """An event fired when task is switched in a CL setup 
        takes a set of named arguments
        """
        if task_num > 0:
            accelerator = kwargs.get("accelerator")
            train_dataloader = kwargs.get("train_dataloader", None)
            assert train_dataloader is not None
            assert self.prev_model is not None
            train_dataloader = accelerator.process_dataloader(train_dataloader)
            # move prev model to the correct device
            self.prev_model = self.prev_model.to(accelerator.root_device)
            freeze_network(self.prev_model)
            if not(self.bg_weighted_ce):
                self.thresholds, self.max_entropy = find_median(
                    train_dataloader,
                    self.nb_current_classes,
                    self.prev_model,
                    accelerator.root_device,
                    accelerator.to_device,
                )

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

    def _preprocess_labels(self, labels, outputs_old):
        """Plop preprocessing step for labels   

        Args:
            labels (torch.float): target segmentation maps
            outputs_old (torch.float): prediction of previous task network

        Returns:
            tuple: a tuple of new labels with pseudo labelled target and adaptive class. factor
        """
        # entropy is the method used by plop paper
        # mask any target having an index lower
        # than the current task lowest class index
        mask_background = labels < self.old_classes
        probs = torch.softmax(outputs_old, dim=1)
        # probs that can be used for prob. based pseudo filtering
        _, pseudo_labels = probs.max(dim=1)

        # the thresholds based on the median computed on the probabilities of the dataset
        mask_valid_pseudo = (entropy(probs) / self.max_entropy) < self.thresholds[
            pseudo_labels
        ]
        # All old labels that are NOT confident enough to be used as pseudo labels:
        labels[~mask_valid_pseudo & mask_background] = 255

        if self.pseudo_ablation is None:
            # All old labels that are confident enough to be used as pseudo labels:
            labels[mask_valid_pseudo & mask_background] = pseudo_labels[
                mask_valid_pseudo & mask_background
            ]
        elif self.pseudo_ablation == "corrected_errors":
            pass  # If used jointly with data_masking=current+old, the labels already
            # contrain the GT, thus all potentials errors were corrected.
        elif self.pseudo_ablation == "removed_errors":
            pseudo_error_mask = labels != pseudo_labels
            kept_pseudo_labels = (
                mask_valid_pseudo & mask_background & ~pseudo_error_mask
            )
            removed_pseudo_labels = (
                mask_valid_pseudo & mask_background & pseudo_error_mask
            )

            labels[kept_pseudo_labels] = pseudo_labels[kept_pseudo_labels]
            labels[removed_pseudo_labels] = 255
        classif_adaptive_factor = 1.0
        if self.classif_adaptive_factor:
            # Number of old/bg pixels that are certain
            num = (mask_valid_pseudo & mask_background).float().sum(dim=(1, 2))
            # Number of old/bg pixels
            den = mask_background.float().sum(dim=(1, 2))
            # If all old/bg pixels are certain the factor is 1 (loss not changed)
            # Else the factor is < 1, i.e. the loss is reduced to avoid
            # giving too much importance to new pixels
            classif_adaptive_factor = num / den
            classif_adaptive_factor = classif_adaptive_factor[:, None, None]
            # Uncertainty based adaptive loss
            classif_adaptive_factor = classif_adaptive_factor.clamp(min=0)
        return labels, classif_adaptive_factor

    def _get_network_output(self, model, img, return_attentions=False):
        """computes network output and return attentions used by plop loss

        Args:
            model (nn.Module): network used to compute attentions
            img (torch.tensor): image input to the network
            return_attentions (bool, optional): flag to return attentions associated with that predictions. Defaults to False.

        Returns:
            tuple: returns tuple of logits and attentions
        """

        attentions = None
        if self.use_logits:
            model.enable_caching_sem_logits()
        logits = model(img, return_attentions=return_attentions,)
        if return_attentions:
            logits, attentions = logits
            if self.use_logits:
                sem_logits = model.pop_sem_logits()
                attentions.append(sem_logits)
        return logits, attentions

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            batch (tuple): a tuple of image and mask
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
        if not(self.bg_weighted_ce):
            logits_new, attentions_new = self._get_network_output(
                model, img, return_attentions=do_distillation
            )
            loss = self.criterion(logits_new, mask).mean()
        else: 
            loss = self.compute_base_loss(
                img, mask, model, train=train, use_weighted_ce=self.bg_weighted_ce and do_distillation, return_attentions=do_distillation
            )
            if do_distillation:
                loss, logits_new, attentions_old, attentions_new, _ = loss 
            else:
                loss, logits_new = loss 
            logits_old = None
            mask_pseudo = None
        if do_distillation:
            # mapping attention
            if not(self.bg_weighted_ce):
                with torch.no_grad():
                    logits_old, attentions_old = self._get_network_output(
                        self.prev_model, img, return_attentions=do_distillation
                    )
                mask_pseudo, classif_adaptive_factor = self._preprocess_labels(
                    mask.clone(), logits_old
                )
                loss = classif_adaptive_factor * self.criterion(logits_new, mask_pseudo,)
                loss = loss.mean()
            # use papers default feature distillation params
            pod_loss = features_distillation(
                attentions_old,
                attentions_new,
                collapse_channels="local",
                labels=mask_pseudo,
                index_new_class=self.old_classes,  # exclude background
                pod_deeplab_mask=False,
                pod_deeplab_mask_factor=None,
                pod_factor=0.01,
                prepro="pow",
                deeplabmask_upscale=True,
                spp_scales=[1, 2, 4],
                pod_options={
                    "switch": {
                        "after": {
                            "extra_channels": "sum",
                            "factor": 0.0005,
                            "type": "local",
                        }
                    }
                },
                outputs_old=logits_old,
                use_pod_schedule=True,
                nb_current_classes=self.nb_current_classes,
                nb_new_classes=self.nb_new_classes,
            )
            return loss + pod_loss, logits_new.argmax(dim=1)
        return (
            loss,
            logits_new.argmax(dim=1),
        )

