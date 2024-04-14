"""Pytorch Lightning wrapper for training segmentation network base model
"""

import pytorch_lightning as pl
import torch

from hydra.utils import instantiate
from visualization import LogMediaQueue, Mode
import math
from .metrics import IoU
from pytorch_lightning.utilities import rank_zero_only


class Model(pl.LightningModule):
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
        super(Model, self).__init__()
        self.network = network
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.scheduler_config = scheduler_config
        self.loss_fn = loss_fn
        self.task_id = task_id
        self.iou = {}
        log_images = "log_images" in train_config and train_config["log_images"]
        log_prototypes = (
            "log_prototypes" in train_config and train_config["log_prototypes"]
        )
        assert not (
            log_images and log_prototypes
        )  # in that case one will starbve and wont find any image in the queue
        self.log_images = log_images or log_prototypes
        # enable testing on OOD
        self.test_ood = False
        self._has_prev_val = False
        self._drift_distance = None

    def _init_scheduler(self, optimizer):
        """Initialize scheduler for given optimizer

        Args:
            optimizer (torch.optim): optimizer used in training

        Returns:
            torch.optim: learning rate scheduler
        """
        if self.scheduler_config is not None:
            scheduler = instantiate(
                self.scheduler_config, optimizer
            )  # add optimizer argument
            scheduler_interval = (
                self.train_config.scheduler_interval
                if "scheduler_interval" in self.train_config
                else "epoch"
            )
            scheduler_frequency = (
                self.train_config.scheduler_frequency
                if "scheduler_frequency" in self.train_config
                else 1
            )
            if hasattr(scheduler, "set_max_iters"):
                scheduler.set_max_iters(self.num_training_steps)
            return {
                "scheduler": scheduler,
                "interval": scheduler_interval,
                "frequency": scheduler_frequency,
            }
        return None

    def _init_optimizer(self, parameters, lr=None):
        """Initialize an optimizer

        Args:
            parameters (generator): network parameters that will be optimized
            lr (float, optional): if given updates the config to use given learning rate. Defaults to None.

        Returns:
            torch.optim: optimizer used for training
        """
        if lr is not None:
            self.optimizer_config.lr = lr
        optimizer = instantiate(self.optimizer_config, parameters)
        return optimizer

    def configure_optimizers(self):
        """configures optimizer and decay scheduler"""
        optimizer = self._init_optimizer(
            self.network.get_parameters(),
            self.train_config.lr_next
            if "lr_next" in self.train_config and self.task_id > 0
            else None,
        )
        if self.scheduler_config is None:
            return optimizer
        scheduler = self._init_scheduler(optimizer)
        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        trainer = self.trainer
        if trainer.max_steps:
            return trainer.max_steps

        limit_batches = trainer.limit_train_batches
        batches = (
            trainer.num_training_batches
            if trainer.num_training_batches > 0
            else len(self.train_dataloader())
        )
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
        if trainer.num_training_batches > 0:
            num_devices = 1
        n_epochs = self.trainer.max_epochs
        batches //= num_devices
        if batches % self.trainer.accumulate_grad_batches == 0:
            batches = batches // self.trainer.accumulate_grad_batches
        else:
            batches = int(math.ceil(batches / self.trainer.accumulate_grad_batches))
        return batches * n_epochs

    @rank_zero_only
    def _init_log_media(self, max_len):
        if self.log_images:
            self.log_media: LogMediaQueue = LogMediaQueue(max_len=max_len)

    def forward(self, x):
        return self.network(x)

    def init_metrics(self, num_classes):
        """Initialize metrics"""
        self.iou["val"] = IoU(num_classes=num_classes).to(self.device)
        self.iou["val_prev"] = IoU(num_classes=num_classes).to(self.device)

    def init_testing_metrics(self, num_classes, n_datasets=1):
        """Initializing testing metrics depending on number of datasets

        Args:
            n_datasets (int, optional): number of testing sets. Defaults to 1.
        """
        for dataset_indx in range(n_datasets):
            self.iou["test.{}".format(dataset_indx)] = IoU(num_classes=num_classes).to(
                self.device
            )

    def _get_prefix(self, base_prefix="train"):
        """get prefix with task number for wandb log

        Args:
            base_prefix (str, optional): base name used. Defaults to "train".

        Returns:
            string: prefix with task number
        """
        if self.task_id >= 0:
            base_prefix += "/Task {}".format(int(self.task_id))
        return base_prefix

    def _log_iou(self, iou_prefix, prefix, detailed_log=False, current_task_only=True):
        """logs iou metrics for dataname/task prefix

        Args:
            iou_prefix (str): prefix used for the iou_metric
            prefix (str): data name for which we compute the IoU
            detailed_log (bool, optional): flag to enable detailed logging of IoU info per class. Defaults to False.
            current_task_only (bool, optional): flag to compute IOU only on classes of current step. Defaults to False.
        """
        iou_metric = self.iou[iou_prefix].to(self.device)
        metrics_avg = iou_metric.compute()
        # add per class IOU metric
        if "prev" in prefix:
            self.trainer.datamodule.task_id -= 1
            current_classes = self.trainer.datamodule.get_current_task_classes()
            self.trainer.datamodule.task_id += 1
        else:
            current_classes = (
                self.trainer.datamodule.get_current_task_classes()
                if current_task_only
                else range(self.trainer.datamodule.get_n_classes())
            )
        if detailed_log:
            # Logs IoU per class in the current task
            for class_indx in current_classes:
                class_name = self.trainer.datamodule.get_label_name(class_indx)
                self.log(
                    "{}/IoU-{}".format(prefix, class_name),
                    metrics_avg.iou_per_class[class_indx].item(),
                )
            self.log(
                "{}/Accuracy".format(prefix),
                IoU.get_mean_per_classes(metrics_avg.accuracy, current_classes),
            )
            self.log(
                "{}/Precision".format(prefix),
                IoU.get_mean_per_classes(metrics_avg.precision, current_classes),
            )
            self.log(
                "{}/Recall".format(prefix),
                IoU.get_mean_per_classes(metrics_avg.recall, current_classes),
            )
            if self.task_id > 0 and not (current_task_only):
                # computes New IoU and IoU on old task
                initial_classes_miou = (
                    metrics_avg.iou_per_class[: self.loss_fn.initial_classes]
                    .mean()
                    .item()
                )
                self.log(
                    "{}/IoU-Old".format(prefix),
                    initial_classes_miou,
                )
                initial_classes_miou_nobg = (
                    metrics_avg.iou_per_class[1 : self.loss_fn.initial_classes]
                    .mean()
                    .item()
                )
                self.log(
                    "{}/IoU-Old-nobg".format(prefix),
                    initial_classes_miou_nobg,
                )
                new_classes_miou = (
                    metrics_avg.iou_per_class[self.loss_fn.initial_classes :]
                    .mean()
                    .item()
                )
                self.log(
                    "{}/IoU-New".format(prefix),
                    new_classes_miou,
                )
        # mIoU for the current task
        self.log(
            prefix + "/mIoU",
            metrics_avg.iou_per_class.mean().item(),
        )
        iou_metric.reset()

    def _log_preds(self, img, target, output_preds, prefix):
        """Function to log information about network preds

        Args:
            img (torch.tensor): set of input images
            target (torch.tensor): set of target masks
            output_preds (torch.tensor): output logits from the model
            prefix (str): prefix name of our network
        """
        if "train" not in prefix:
            self.iou[prefix] = self.iou[prefix].to(self.device)
            self.iou[prefix](output_preds, target)
        if self.task_id >= 0:
            self.log(
                "task",
                self.task_id,
                prog_bar=True,
            )
        if self.log_images:
            # split is used for the case where we have multiple test datasets
            self._append_images(prefix, img, target, output_preds)

    @rank_zero_only
    def _append_images(self, prefix, img, target, output_preds):
        media_prefix = Mode.TRAIN
        if "val" in prefix:
            media_prefix = Mode.VAL
        elif "test" in prefix:
            media_prefix = Mode.TEST
        self.log_media.append(
            {"inputs": img, "labels": target, "preds": output_preds}, media_prefix
        )

    def step(self, batch, batch_idx, prefix="train", optimizer_idx=None):
        """step on batch used to compute loss value and predictions

        Args:
            batch (tuple): tuple of images, target masks and task id in case of CL
            prefix (str, optional): dataname / task num used for logging. Defaults to "train".

        Returns:
            dict: dictionary of computed metrics including loss
        """
        batch = self.loss_fn.preprocess_batch(batch)
        # set loss fn device
        self.loss_fn.set_device(self.device)
        # call on_batch_start
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        if self.trainer.max_epochs is None:
            epoch = self.global_step
            max_epochs = self.trainer.max_steps
        self.loss_fn.on_train_batch_start(
            epoch=epoch,
            max_epochs=max_epochs,
            batch_idx=batch_idx,
        )
        loss, output_mask = self.loss_fn.compute_loss(
            batch, self.network, train="train" in prefix
        )
        if isinstance(batch, dict):
            batch = batch["main"]
        # loss function returns output mask
        self._log_preds(batch[0], batch[1], output_mask, prefix)
        return {"loss": loss}

    def end(self, outputs, prefix="train"):
        """Aggregates stats from each validation step
        Args:
            outputs (list): list of stats from each validation step call
        Returns:
            dict: aggregated stats for the validation
        """
        prefix = self._get_prefix(prefix)
        if len(outputs) == 0:
            return {prefix + "_loss": -1 * self.global_step}

        metric_keys = outputs[0].keys()
        output_logs = {}
        for metric_name in metric_keys:
            key_name = prefix + "/" + metric_name
            all_values = []
            for x in outputs:
                if metric_name in x:
                    all_values.append(x[metric_name])
            if len(all_values) == 0:
                continue
            avg_value = torch.stack(all_values).mean()
            output_logs[key_name] = avg_value
        if "test" in prefix:
            output_logs[prefix + "/num_params_million"] = (
                torch.tensor(
                    [p.numel() for p in self.network.parameters() if p.requires_grad]
                ).sum()
                / 10e5
            )
        return output_logs, prefix

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Computes training step

        Args:
            batch (tuple): tuple of images, target masks and task id in case of CL
            batch_nb (int): batch number
            optimizer_idx (int): index of optimizer used in case of multi-optimizers

        Returns:
            torch.tensor: loss value
        """
        train_step_metrics = self.step(batch, batch_idx, "train", optimizer_idx)
        if self._drift_distance is not None:
            prefix = self._get_prefix("train")
            train_step_metrics["representation_drift"] = self._drift_distance
        return train_step_metrics

    def training_epoch_end(self, outputs):
        """Compute and log metrics across epoch

        Args:
            outputs (list): list of outputs computed at each step
        """
        metrics, prefix = self.end(outputs, prefix="train")
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        """Computes validation step

        Args:
            batch (tuple): tuple of images, target masks and task id in case of CL
            batch_idx (int): batch number

        Returns:
            torch.tensor: loss value
        """
        if isinstance(batch, dict):
            metrics = self.step(batch["current"], batch_idx, "val")
            self.log(
                "val_loss",
                metrics["loss"].item(),
            )
            self.step(batch["prev"], batch_idx, "val_prev")
            self._has_prev_val = True
            return metrics
        self._has_prev_val = False
        metrics = self.step(batch, batch_idx, "val")
        self.log(
            "val_loss",
            metrics["loss"].item(),
        )
        return metrics

    def validation_epoch_end(self, outputs):
        """Aggregates stats from each validation step
        Args:
            outputs (list): list of stats from each validation step call
        Returns:
            dict: aggregated stats for the validation
        """
        if self._has_prev_val:
            _, prefix = self.end(outputs, prefix="val_prev")
            self._log_iou("val_prev", prefix, detailed_log=True, current_task_only=True)
        metrics, prefix = self.end(outputs, prefix="val")
        self.log_dict(metrics)
        self._log_iou("val", prefix, detailed_log=True, current_task_only=True)

    def test_step(self, batch, batch_idx, dataset_idx=0):
        """Computes test step

        Args:
            batch (tuple): tuple of images, target masks and task id in case of CL
            batch_idx (int): batch number
            dataset_idx (int): dataset index in case of multiple test sets

        Returns:
            torch.tensor: loss value
        """
        test_step_metrics = self.step(
            batch, batch_idx, "{}.{}".format("test", dataset_idx)
        )
        return test_step_metrics

    def test_epoch_end(self, outputs):
        """Aggregates stats from each test step
        Args:
            outputs (list): list of stats from each validation step call
        Returns:
            dict: aggregated stats for the test
        """
        n_datasets = 1
        if type(outputs[0]) is list:
            n_datasets = len(outputs)
        for dataset_idx in range(n_datasets):
            if n_datasets > 1:
                metrics, prefix = self.end(
                    outputs[dataset_idx], prefix="{}.{}".format("test", dataset_idx)
                )
            else:
                metrics, prefix = self.end(
                    outputs, prefix="{}.{}".format("test", dataset_idx)
                )
            self.log_dict(metrics)
            self._log_iou(
                "{}.{}".format("test", dataset_idx),
                prefix,
                detailed_log=True,
                current_task_only=False,
            )

    def log_dict(self, input_dict):
        """takes dictionary of metrics and logs to pt default logger

        Args:
            input_dict (dict): metric dictionary
        """
        for name, value in input_dict.items():
            self.log(
                name,
                value.item(),
            )
