"""Pytorch Lightning wrapper for training both ood aux and actual clf
"""

from .model import Model
from .metrics import IoU
from segmentation_models_pytorch.losses import FocalLoss
import torch
from sklearn.metrics import f1_score, accuracy_score


class OODModel(Model):
    def __init__(
        self,
        network,
        optimizer_config,
        train_config,
        loss_fn,
        scheduler_config=None,
        task_id=-1,
    ):
        """init segmentation model

        Args:
            network (nn.module): model used for segmentation from models folder
            optimizer (dict ): optimizer configuration
            loss_fn (.loss.base_loss): custom loss function used for the problem
            scheduler (dict): scheduler configuration. Defaults to None
            task_id (int): step number in cl setup. Defaults to -1 for joint training
        """
        super().__init__(
            network,
            optimizer_config,
            train_config,
            loss_fn,
            scheduler_config=scheduler_config,
            task_id=task_id,
        )
        self.siamese_loss = FocalLoss(
            mode="binary", ignore_index=self.loss_fn.ignore_index, gamma=2
        )

    def init_metrics(self, num_classes):
        """Initialize metrics"""
        super().init_metrics(num_classes)
        # seen / unseen class
        self.iou["val_aux_bg"] = IoU(
            num_classes=2,
        ).to(self.device)

    def init_testing_metrics(self, num_classes, n_datasets=1):
        """Initializing testing metrics depending on number of datasets

        Args:
            n_datasets (int, optional): number of testing sets. Defaults to 1.
        """
        super().init_testing_metrics(num_classes, n_datasets)
        # Used to test aux bg classifier performance
        for dataset_indx in range(n_datasets):
            self.iou["test.{}_aux_bg".format(dataset_indx)] = IoU(
                num_classes=2,
            ).to(self.device)

    def _log_iou(self, iou_prefix, prefix, detailed_log=False, current_task_only=True):
        """logs iou metrics for dataname/task prefix

        Args:
            iou_prefix (str): prefix used for the iou_metric
            prefix (str): data name for which we compute the IoU
            detailed_log (bool, optional): flag to enable detailed logging of IoU info per class. Defaults to False.
            current_task_only (bool, optional): flag to compute IOU only on classes of current step. Defaults to False.
        """
        super()._log_iou(iou_prefix, prefix, True, current_task_only)
        if "train" in prefix or "prev" in prefix:
            return
        # now computing iou for aux network
        # maybe check if the prefix exists
        iou_prefix = "{}_aux_bg".format(iou_prefix)
        prefix = "{}_aux_bg".format(prefix)
        iou_metric = self.iou[iou_prefix].to(self.device)
        metrics_avg = iou_metric.compute()
        # Logs IoU per class in the current task
        for class_indx, class_name in zip(range(2), ["bg", "not-bg"]):
            self.log(
                "{}/IoU-{}".format(prefix, class_name),
                metrics_avg.iou_per_class[class_indx].item(),
            )
        self.log(
            "{}/Accuracy".format(prefix),
            IoU.get_mean_per_classes(metrics_avg.accuracy, range(2)),
        )
        self.log(
            "{}/Precision".format(prefix),
            IoU.get_mean_per_classes(metrics_avg.precision, range(2)),
        )
        self.log(
            "{}/Recall".format(prefix),
            IoU.get_mean_per_classes(metrics_avg.recall, range(2)),
        )
        # mIoU for the current task
        self.log(prefix + "/mIoU", metrics_avg.iou_per_class.mean().item())
        iou_metric.reset()

    def _log_aux_probs(self, name, prefix, logits):

        self.log(
            "{}/{}_prob_mean".format(prefix, name),
            logits.mean().item(),
        )
        self.log(
            "{}/{}_prob_var".format(prefix, name),
            logits.var().item(),
        )

    def step(self, batch, batch_idx, prefix="train", optimizer_idx=None):
        """step on batch used to compute loss value and predictions

        Args:
            batch (tuple): tuple of images, target masks and task id in case of CL
            prefix (str, optional): dataname / task num used for logging. Defaults to "train".

        Returns:
            dict: dictionary of computed metrics including loss
        """
        loss = super().step(
            batch, batch_idx=batch_idx, prefix=prefix, optimizer_idx=optimizer_idx
        )["loss"]
        batch = self.loss_fn.preprocess_batch(batch)
        if isinstance(batch, dict):
            batch = batch["main"]
        img = batch[0]
        mask = batch[1]
        if (
            "train" in prefix
            or "prev" in prefix
            or not ((mask == 0).any() and (mask != 0).any())
        ):
            return {"loss": loss}
        new_target = mask.clone()
        # 1 for fg, 0 for bg
        new_target[(mask != 0) & (mask != self.loss_fn.ignore_index)] = 1
        new_target[(mask == 0)] = 0
        penultimate_output = self.network.get_penultimate_output(img)
        seen_prob = self.network.seen_fg_network.get_seen_probs(
            penultimate_output,
            self.loss_fn.prototypes,
        ).max(1)[0]
        proto_preds = (seen_prob > 0.5).float().squeeze(dim=1)
        aux_prefix = "{}_aux_bg".format(prefix)
        self.iou[aux_prefix] = self.iou[aux_prefix].to(self.device)
        self.iou[aux_prefix].update(proto_preds, new_target)
        seen_prob = seen_prob.squeeze(dim=1)
        prefix = self._get_prefix(prefix)
        self._log_aux_probs("bg", prefix, seen_prob[new_target == 0])
        self._log_aux_probs(
            "fg",
            prefix,
            seen_prob[new_target == 1],
        )
        old_cl, nb_cl = self.loss_fn.old_classes, self.loss_fn.nb_current_classes
        # Analyze current task output
        if ((mask >= old_cl) & (mask <= nb_cl)).bool().any():
            self._log_aux_probs(
                "fg_current",
                prefix,
                seen_prob[(new_target == 1) & (mask >= old_cl) & (mask <= nb_cl)],
            )
        if self.loss_fn.old_classes > 0:
            self._log_aux_probs(
                "old_cl",
                prefix,
                seen_prob[(new_target != 0) & (new_target <= self.loss_fn.old_classes)],
            )
        return {"loss": loss}

    def test_step(self, batch, batch_idx, dataset_idx=0):
        """Computes test step

        Args:
            batch (tuple): tuple of images, target masks and task id in case of CL
            batch_idx (int): batch number
            dataset_idx (int): dataset index in case of multiple test sets

        Returns:
            torch.tensor: loss value
        """
        if self.test_ood:
            img = batch[0].float()
            penultimate_output = self.network.get_penultimate_output(img)
            seen_prob = self.network.seen_fg_network.forward_seen_before(
                penultimate_output,
                self.loss_fn.prototypes,
            ).sigmoid()[:, -1, :, :]
            proto_preds = (seen_prob > 0.5).float().squeeze(dim=1)
            # now computing OOD metrics
            prefix = "test.{}_aux_ood".format(dataset_idx)
            targets = torch.zeros_like(proto_preds).long()
            self._log_aux_probs("ood_detection", prefix, seen_prob[targets == 0])
            y_true, y_preds = (
                targets.clone().reshape(-1).cpu(),
                (seen_prob > 0.5).clone().reshape(-1).cpu(),
            )
            current_f1 = f1_score(y_true, y_preds, pos_label=0)
            self.log("{}/{}_current".format(prefix, "F1_score"), current_f1)
            current_acc = accuracy_score(y_true, y_preds)
            self.log("{}/{}_current".format(prefix, "Accuracy"), current_acc)
            return {
                "F1_score": torch.tensor(current_f1).type_as(seen_prob),
                "verification_acc": torch.tensor(current_acc).type_as(seen_prob),
            }
        return super().test_step(batch, batch_idx, dataset_idx)
