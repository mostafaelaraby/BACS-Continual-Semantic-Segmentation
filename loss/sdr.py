import torch.nn.functional as F
from .prototypes import Prototypes
from training.utils import freeze_network
from training.loss_utils import UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy
import torch
from tqdm import tqdm
import torch.nn as nn


class SDR(Prototypes):
    """Basic CrossEntropy loss used for finetuning baseline"""

    def __init__(
        self,
        name="SDR",
        ignore_index=255,
        lfc_sep_clust=1e-3,
        loss_fc=1e-3,
        loss_featspars=1e-3,
        loss_de_prototypes=0.01,
        loss_kd=100,
    ):
        """initialize a new loss

        Args:
            name (str): name of the loss function
            ignore_index (int): index of the ignore pixel
        """
        super().__init__(name, ignore_index)
        self.prev_model = None
        self.lfc_sep_clust = lfc_sep_clust
        self.loss_fc = loss_fc
        self.loss_featspars = loss_featspars
        self.loss_de_prototypes = loss_de_prototypes
        self.loss_kd = loss_kd
        self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=1.0)
        self.criterion = None
        self.skip_updating_bg = False
        self.not_sequential_mode = True
        self.use_distillation = False

    def on_train_start(self, task_num, **kwargs):
        """An event fired when task is switched in a CL setup
        takes a set of named arguments
        """
        accelerator = kwargs.get("accelerator")
        if task_num > 0:
            assert self.prev_model is not None
            freeze_network(self.prev_model)
            # move prev model to the correct device
            self.prev_model = self.prev_model.to(accelerator.root_device)
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes,
                ignore_index=self.ignore_index,
                reduction="mean",
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index, reduction="mean"
            )
        model = kwargs.get("model")
        penultimate_dim = model.get_penultimate_layer_dim()
        self._init_prototypes(task_num, accelerator, penultimate_dim)
        self.not_sequential_mode = kwargs.get("not_sequential_mode")
        self.use_distillation = task_num > 0
        self.skip_updating_bg = self.not_sequential_mode and task_num > 0

    def on_train_end(self, **kwargs):
        """An event fired at end of training to cache the network of previous task"""
        super().on_train_end(**kwargs)
        pre_last_tasks = kwargs.get("pre_last_tasks")
        if not (pre_last_tasks):
            return
        model = kwargs.get("model") if "model" in kwargs else None
        if model is not None:
            self.prev_model = model.clone()
            freeze_network(self.prev_model)

    def _init_prototypes(self, task_num, accelerator, penultimate_dim):
        """Initializes the prototypes at start of each task

        Args:
            task_num (int): task number
            accelerator (Trainer.accelerator): pytorch lightning accelarator used
            penultimate_dim (int): dimension size of penultimate layer
        """
        # we have a single prototype per task explaining its representations
        if task_num > 0:
            new_prototypes = torch.zeros(
                [self.nb_new_classes, penultimate_dim],
                device=accelerator.root_device,
                requires_grad=False,
            )
            new_count_features = torch.zeros(
                [self.nb_new_classes],
                device=accelerator.root_device,
                requires_grad=False,
            )
            self._prototypes_tensors = torch.cat(
                [self._prototypes_tensors, new_prototypes], dim=0
            )
            self._count_features = torch.cat(
                [self._count_features, new_count_features], dim=0
            )
        else:
            self._prototypes_tensors = torch.zeros(
                [self.nb_current_classes, penultimate_dim],
                device=accelerator.root_device,
                requires_grad=False,
            )
            self._count_features = torch.zeros(
                [self.nb_current_classes],
                dtype=torch.long,
                device=accelerator.root_device,
                requires_grad=False,
            )
        self._prototypes_tensors.requires_grad = False
        self._count_features.requires_grad = False

    def update_prototypes(self, model, img, target):
        """Updates the prototypes based on current batchd data

        Args:
            model (networks.basemodel): input training model
            img (nn.Tensor): tensor of images in current batch
            target (nn.Tensor): tensor of corresponding targets
            current_weight (int): weight given to current task
        """
        features = model.get_penultimate_output(img).detach()
        extracted_indx_proto = self._extract_labels_prototype_index(features, target)
        if extracted_indx_proto is None:
            return None
        (
            features,
            feat_dim,
            labels_down,
            _,
        ) = extracted_indx_proto
        cl_present = torch.unique(input=labels_down)
        if cl_present[-1] == self.ignore_index:
            cl_present = cl_present[:-1]
        # overlap / disjoint exclude updating the background in incremental steps
        if self.skip_updating_bg and cl_present[0] == 0:
            # remove the background in both overlap/disjoint
            cl_present = cl_present[1:]

        for cl in cl_present:
            masked_cl = (labels_down == cl).bool()
            n_features = masked_cl.sum()
            features_cl = features[masked_cl.expand(-1, feat_dim, -1, -1)].view(
                features.shape[1], -1
            )
            features_cl_sum = torch.sum(features_cl, dim=-1)
            features_running_mean_tot_cl = (
                features_cl_sum
                + self._count_features[cl] * self._prototypes_tensors[cl]
            ) / (self._count_features[cl] + n_features)
            self._count_features[cl] += n_features
            self._prototypes_tensors[cl] = features_running_mean_tot_cl

    def feature_clustering_separation(self, mask, features):
        loss_features_clustering = torch.tensor(0.0, device=self.device)
        loss_separationclustering = torch.tensor(0.0, device=self.device)
        labels_down = (
            F.interpolate(
                input=mask.clone().unsqueeze(dim=1).double(),
                size=(features.shape[2], features.shape[3]),
                mode="nearest",
            )
        ).long()
        cl_present = torch.unique(input=labels_down)
        if cl_present[-1] == self.ignore_index:
            cl_present = cl_present[:-1]
        features_local_mean = torch.zeros(
            [self.nb_current_classes, features.shape[1]], device=self.device
        )
        for cl in cl_present:
            features_cl = features[
                (labels_down == cl).expand(-1, features.shape[1], -1, -1)
            ].view(features.shape[1], -1)
            features_local_mean[cl] = torch.mean(features_cl, dim=-1)
            loss_to_use = nn.MSELoss()
            loss_features_clustering += loss_to_use(
                features_cl,
                self.prototypes[cl].unsqueeze(1).expand(-1, features_cl.shape[1]),
            )
            loss_features_clustering /= cl_present.shape[0]
        features_local_mean_reduced = features_local_mean[
            cl_present, :
        ]  # remove zero rows
        inv_pairwise_D = (
            1
            / torch.cdist(
                features_local_mean_reduced.unsqueeze(dim=0),
                features_local_mean_reduced.unsqueeze(dim=0),
            ).squeeze()
        )
        loss_separationclustering_temp = inv_pairwise_D[
            ~torch.isinf(inv_pairwise_D)
        ].mean()
        if ~torch.isnan(loss_separationclustering_temp):
            loss_separationclustering = loss_separationclustering_temp
        loss_separationclustering *= self.lfc_sep_clust
        loss_features_clustering *= self.loss_fc
        if torch.isnan(loss_features_clustering):
            loss_features_clustering = torch.tensor(0.0)
        return loss_features_clustering + loss_separationclustering

    def feature_sparsification_loss(
        self,
        mask,
        features,
    ):
        eps = 1e-15
        feature_sparsification_loss = torch.tensor(0.0, device=self.device)
        labels = mask.unsqueeze(dim=1)
        labels_down = (
            F.interpolate(
                input=labels.double(),
                size=(features.shape[2], features.shape[3]),
                mode="nearest",
            )
        ).long()
        features_norm = torch.zeros_like(features)
        classes = torch.unique(labels_down)
        if classes[-1] == 0:
            classes = classes[:-1]
        for cl in classes:
            cl_mask = labels_down == cl
            features_norm += (
                features
                / (
                    torch.max(features[cl_mask.expand(-1, features.shape[1], -1, -1)])
                    + eps
                )
            ) * cl_mask.float()

        if features_norm.sum() > 0:
            shrinked_value = torch.sum(torch.exp(features_norm), dim=1, keepdim=True)
            summed_value = torch.sum(features_norm, dim=1, keepdim=True)
            feature_sparsification_loss = shrinked_value / (summed_value + eps)
        return self.loss_featspars * feature_sparsification_loss.mean()

    def distillation_prototypes_loss(self, outputs, outputs_old, features, mask):
        # loss_de_prototypes_sumafter=True
        outputs = torch.tensor(0.0, device=self.device)
        MSEloss_to_use = nn.MSELoss()
        labels = mask.unsqueeze(dim=1)
        labels_down = (
            F.interpolate(
                input=labels.double(),
                size=(features.shape[2], features.shape[3]),
                mode="nearest",
            )
        ).long()
        labels_down_bgr_mask = (labels_down == 0).long()
        if not (self.not_sequential_mode):
            # sequential mode
            pseudolabel_old_down = labels_down * (labels_down < self.old_classes).long()
        else:
            outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
            outputs_old_down = (
                F.interpolate(
                    input=outputs_old.double(),
                    size=(features.shape[2], features.shape[3]),
                    mode="nearest",
                )
            ).long()
            pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
        cl_present = torch.unique(input=pseudolabel_old_down).long()
        if cl_present[0] == 0:
            cl_present = cl_present[1:]
        for cl in cl_present:
            prototype = self.prototypes.detach()[cl]
            current_features = features[
                (pseudolabel_old_down == cl).expand_as(features)
            ].view(-1, features.shape[1])
            current_proto = torch.mean(current_features, dim=0)
            outputs += MSEloss_to_use(current_proto, prototype) / cl_present.shape[0]
        return self.loss_de_prototypes * outputs

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            batch (tuple): tuple of image and mask
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
        # now we start with default cross-entropy loss
        preds_mask, penultimate_output = model(img, return_penultimate=True)
        loss = self.criterion(preds_mask, mask)
        if train and self.use_distillation:
            # update prototypes first for training only
            self.update_prototypes(model, img, mask)
            loss += self.feature_sparsification_loss(mask, penultimate_output)
            loss += self.feature_clustering_separation(mask, penultimate_output)
            # use loss of prototypes distillation
            with torch.no_grad():
                preds_mask_old = self.prev_model(
                    img,
                )
            loss += self.distillation_prototypes_loss(
                preds_mask,
                preds_mask_old,
                penultimate_output,
                mask,
            )
            loss += self.loss_kd * self.lkd_loss(preds_mask, preds_mask_old)
        preds_mask = preds_mask.argmax(dim=1)
        return loss, preds_mask
