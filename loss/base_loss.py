import torch.nn.functional as F
import torch
from training.loss_utils import WeightedCrossEntropy, UnbiasedKnowledgeDistillationLoss
from segmentation_models_pytorch.losses import FocalLoss
import numpy as np
import math


class BaseLoss:
    """Parent class for losses"""

    def __init__(self, name, ignore_index=255):
        """initialize a new loss

        Args:
            name (str): name of the loss function
            ignore_index (int): index of the ignore pixel
        """
        self.name = name
        # number of classes seen in previous tasks CL
        self.old_classes = 0
        # number of classes in first task
        self.initial_classes = 0
        # number of classes to inrement per task
        self.increment = 0
        # total number of classes in current task
        self.nb_current_classes = 0
        # number of classes added to current new task which is increment
        self.nb_new_classes = 0
        # current epoch number
        self.epoch_number = 0
        # max number of epochs
        self.max_epochs = 0
        # device on which training is done
        self.device = None
        # ignore index used by the dataset
        self.ignore_index = ignore_index
        # prototype computation if enabled
        self._prototypes = None
        # accelerator used
        self.accelerator = None
        # seen detector focal loss
        self.seen_fgloss = None
        self.init_seen_focal_loss()
        # same task sampled in replay buffer
        self.same_task = False
        self.last_task = False
        self.first_task = True
        # init weighted loss
        self.weighted_ce = None
        # previous model
        self.prev_model = None
        self.init_weighted_loss()

    def set_device(self, device):
        """sets training device

        Args:
            device : device used for training
        """
        self.device = device

    def init_seen_focal_loss(self, gamma=2, alpha=None):
        self.seen_fgloss = FocalLoss(
            mode="binary",
            ignore_index=self.ignore_index,
            gamma=gamma,
            alpha=alpha,
            reduction="mean",
        )

    def init_weighted_loss(self, gamma=2, threshold=0.5, ukd=True):
        # weighted ce loss
        self.weighted_ce = WeightedCrossEntropy(
            ignore_index=self.ignore_index, gamma=gamma, threshold=threshold, ukd=ukd
        )
        self.weighted_ce.base_loss = self
        self.lkd_loss = UnbiasedKnowledgeDistillationLoss()

    def _update_task(self, task_num):
        """Update task information

        Args:
            task_num (int): takes zero based task number as input to update nb_classes
        """
        self.nb_new_classes = self.increment
        self.old_classes = self.get_n_old_classes(task_num)
        self.nb_current_classes = self.initial_classes + self.increment * task_num
        self.first_task = task_num == 0

    def get_n_old_classes(self, task_num):
        return (
            self.initial_classes + self.increment * (task_num - 1)
            if task_num > 0
            else 0
        )

    def label_to_task_num(self, label):
        current_task = 0
        if self.increment > 0:
            # to move label index from 0 based to 1 based
            if hasattr(label, "cpu"):
                label = label.cpu().numpy()
            current_task = (label + 1 - self.initial_classes) / self.increment
            current_task[current_task < 0] = 0
            current_task = np.rint(current_task)
        return current_task

    def set_continual_task_size(self, initial_classes, increment=0):
        """sets the initial classes and increment for continual learning

        Args:
            initial_classes (int): number of classes in first task
            increment (int, optional): number of new classes added in new tasks. Defaults to 0.
        """
        if self._prototypes is not None:
            self._prototypes.set_continual_task_size(initial_classes, increment)
        self.initial_classes = initial_classes
        self.increment = increment
        self.nb_current_classes = self.initial_classes

    def init_prototype_compute(self):
        from .prototypes import Prototypes

        self._prototypes = Prototypes(
            name="Prototype_{}".format(self.name),
            ignore_index=self.ignore_index,
        )

    @property
    def prototypes(self):
        return self._prototypes._prototypes_tensors

    def are_prototypes_ready(self):
        """Detects if at least a data point per class seen so far

        Returns:
            bool: flag to denote if all our prototypes are nonzeros
        """
        if self._prototypes is not None:
            return self._prototypes.are_prototypes_ready()
        return False

    def on_fit_start(self, task_num, **kwargs):
        """Event fired when fit starts

        Args:
            task_num (int): task number
        """
        self._update_task(task_num)
        if self._prototypes is not None:
            self._prototypes.on_fit_start(task_num, **kwargs)
            self._prototypes.on_train_start(task_num, **kwargs)

    def on_train_start(self, task_num, **kwargs):
        """An event fired at start of each step with old network passed
        Args:
            task_num (int): takes zero based task number as input to update nb_classes
        """
        pass

    def on_train_end(self, **kwargs):
        """An event fired at end of training to cache the network of previous task"""
        if self._prototypes is not None:
            self._prototypes.on_train_end(**kwargs)

    def on_train_batch_start(self, **kwargs):
        """An event fired at start of each step"""
        self.epoch_number = kwargs.get("epoch")
        self.max_epochs = kwargs.get("max_epochs")

    def compute_base_loss(
        self,
        img,
        mask,
        model,
        task_num=-1,
        weights=None,
        train=True,
        use_weighted_ce=False,
        return_attentions=False,
    ):
        """Base cross entropy loss

        Args:
            preds (Tensor): preds logits from the network
            mask (Tensor): true labels
            model (BaseNetwork): model used for inference
            task_num (int): index of the task used in computation for the proto-selection
            weights (Tensor): per class weight
            train (bool): in case of train mode

        Returns:
            tuple(torch.tensor, torch.tensor): loss function value, output predictions
        """
        old_atts, attentions, seen_prob = (None, None, None)
        is_experience_replay = task_num != -1
        self.weighted_ce.old_cl = self.old_classes
        # only trained on first task
        train_seen_detector = (
            model.seen_fg_network is not None
            and (self.same_task or not (is_experience_replay))
            and train
        )

        return_penultimate = (
            train_seen_detector or use_weighted_ce or self._prototypes is not None
        )
        preds_mask, penultimate_output, attentions = model(
            img,
            return_penultimate=True,
            return_attentions=True,
        )
        if self.prev_model is not None and train and return_attentions:
            # make the model return an object including everything needed ;)
            _, _, old_atts = self.prev_model(
                img, return_penultimate=True, return_attentions=True
            )
        if return_penultimate and train:
            self._prototypes.update_feats_prototypes(penultimate_output, mask)
        are_proto_ready = self.are_prototypes_ready()
        train_seen_detector = train_seen_detector and are_proto_ready
        if use_weighted_ce and train:
            with torch.no_grad():
                revert_to_train = False
                if model.training:
                    model.eval()
                    revert_to_train = True
                seen_prob = model.seen_fg_network.get_seen_probs(
                    penultimate_output, self.prototypes, bg_detect=True
                ).detach()
                if revert_to_train:
                    model.train()
            if task_num == -1:
                task_num = self.prototypes.shape[0] - 1
            loss = self.weighted_ce(preds_mask, mask, seen_prob, task_num)
        else:
            loss = F.cross_entropy(
                preds_mask, mask, ignore_index=self.ignore_index, weight=weights
            )
        if train_seen_detector:
            seen_detector_weight = max(
                0, 1 - np.exp((self.epoch_number - self.max_epochs))
            )
            loss += seen_detector_weight * self._compute_seen_fg_loss(
                penultimate_output,
                mask,
                model,
                task_num=task_num,
            )
        if return_attentions and train:
            return loss, preds_mask, old_atts, attentions, seen_prob
        return loss, preds_mask

    def _compute_seen_fg_loss(self, penulimate_output, mask, model, task_num=-1):
        new_target = mask.clone()
        # 1 for fg, 0 for bg
        new_target[(mask != 0) & (mask != self.ignore_index)] = 1
        new_target[(mask == 0)] = 0
        if not ((mask == 0).any()):
            # skip batches not having background in them
            return 0
        if task_num == -1:
            task_num = self.prototypes.shape[0] - 1
        model.seen_fg_network.set_stop_gradients(not (self.first_task))
        seen_before_logits = model.seen_fg_network.get_seen_map_task(
            penulimate_output,
            self.prototypes,
            task_num,
        )
        loss = self.seen_fgloss(seen_before_logits, new_target.unsqueeze(1).float())
        return loss

    def preprocess_batch(self, batch):
        if isinstance(batch, dict):
            for key in batch:
                batch[key][0] = batch[key][0].float()
                batch[key][1] = batch[key][1].long()
        else:
            batch[0] = batch[0].float()
            batch[1] = batch[1].long()
        return batch

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            batch (tuple(tensor, tensor)): tuple of image and mask
            model (networks.BaseNetwork): model used for training
            train (bool): flag denoting this is train or val/test

        Raises:
            NotImplementedError: needs to be overriden in child classes
        """
        pass
