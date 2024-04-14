# adapted from https://github.com/Shreeyak/pytorch-lightning-segmentation-template/blob/master/seg_lapa/callbacks/log_media.py
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pytorch_lightning.loggers.base import LoggerCollection
import torch
import wandb
from pytorch_lightning import loggers as pl_loggers 
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only

# Project-specific imports for logging media to disk
import cv2
import math 
import numpy as np
from PIL import Image
from .base_medialogger import BaseMediaLogger, Mode, LogMediaQueue
from torchvision import transforms

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [1/s for s in self.std]),
                                transforms.Normalize(mean = [ -1 * m for m in self.mean ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return self.inv_trans(tensor)


@dataclass
class PredData:
    """Holds the data read and converted from the LightningModule's LogMediaQueue"""

    inputs: np.ndarray
    labels: np.ndarray
    preds: np.ndarray

class LogMedia(BaseMediaLogger):
    r"""Logs model output images and other media to weights and biases

    This callback required adding an attribute to the LightningModule called ``self.log_media``. This is a circular
    queue that holds the latest N batches. This callback fetches the latest data from the queue for logging.

    Usage:
        import pytorch_lightning as pl

        class MyModel(pl.LightningModule):
            self.log_media: LogMediaQueue = LogMediaQueue(max_len)

        trainer = pl.Trainer(callbacks=[LogMedia()])

    Args:
        period_epoch (int): If > 0, log every N epochs
        period_step (int): If > 0, log every N steps (i.e. batches)
        max_samples (int): Max number of data samples to log
        save_to_disk (bool): If True, save results to disk
        save_latest_only (only): If True, will overwrite prev results at each period.
        exp_dir (str or Path): Path to directory where results will be saved
        verbose (bool): verbosity mode. Default: ``True``.
    """

    def __init__(
        self,
        datamodule,
        max_samples: int = 10,
        period_epoch: int = 1,
        period_step: int = 0,
        save_to_disk: bool = True,
        save_latest_only: bool = True,
        exp_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__(
            datamodule,
            max_samples,
            period_epoch,
            period_step,
            save_to_disk,
            save_latest_only,
            exp_dir,
            verbose,
        )
        self.valid_logger = False
        # denormalize
        normalization_mean_std = datamodule._get_normalization_mean_std()
        self.denormalizer = (
            UnNormalize(normalization_mean_std[0], normalization_mean_std[1])
            if normalization_mean_std is not None
            else None
        )
        self.saved_batch = {}
    
    def setup(self, trainer, pl_module, stage: str):
        super().setup(trainer, pl_module, stage)
        # get saved batch that will be plotted 
        prefix = "train"
        if "val" in stage:
            prefix="val"
        elif "test" in stage:
            prefix="test"
        current_batch = trainer.datamodule.get_common_batch(prefix)
        self.saved_batch[prefix] = {"inputs": current_batch[0], "labels": current_batch[1]}

    def map_color_palette(self, label) -> np.ndarray:
        """Generates RGB visualization of label by applying a color palette
            Label should contain an uint class index per pixel.
            Args:
                Args:
                label (numpy.ndarray): Each pixel has uint value corresponding to it's class index
                                    Shape: (H, W), dtype: np.uint8, np.uint16
                palette : Which color palette to use.
            Returns:
                numpy.ndarray: RGB image, with each class mapped to a unique color.
                            Shape: (H, W, 3), dtype: np.uint8
        """
        # remove ignored pixel before plotting 
        label[label == 255] = 0
        if len(self.class_colors) < label.max():
            raise ValueError(
                f"The chosen color palette has only {len(self.class_colors)} values. It does not have"
                f" enough unique colors to represent all the values in the label ({label.max()})"
            )

        # Map grayscale image's pixel values to RGB color palette
        _im = Image.fromarray(label)
        _im.putpalette(self.class_colors)
        _im = _im.convert(mode="RGB")
        im = np.asarray(_im)
        return im

    def _log_results(
        self, trainer, pl_module, mode: Mode, batch_idx: Optional[int] = None
    ):
        pred_data = self._get_preds_from_lightningmodule(pl_module, mode)
        self._save_media_to_logger(trainer, pred_data, mode)
        self._save_media_to_disk(trainer, pred_data, mode, batch_idx)

    
    def _get_preds_from_lightningmodule(
        self, pl_module, mode: Mode
    ) -> Optional[PredData]:
        """Fetch latest N batches from the data queue in LightningModule.
        Process the tensors as required (example, convert to numpy arrays and scale)
        """
        if mode.value not in self.saved_batch:  # Empty queue
            rank_zero_warn(
                f"\nEmpty LogMediaQueue! Mode: {mode}. Epoch: {pl_module.trainer.current_epoch}"
            )
            return None

        media_data = self.saved_batch[mode.value]
        device_id = next(iter(pl_module.network.parameters())).device
        with torch.no_grad():
            output_preds = pl_module.network(media_data["inputs"].to(device_id)).argmax(1).detach().cpu()
            media_data["preds"] = output_preds
        if self.denormalizer is not None:
            inputs =self.denormalizer(media_data["inputs"]) 
        else:
            inputs = media_data["inputs"] 
        labels = media_data["labels"] 
        preds = media_data["preds"] 

        # Limit the num of samples and convert to numpy
        inputs = (
            inputs[: self.max_samples].detach().cpu().numpy().transpose((0, 2, 3, 1))
        )
        inputs = (inputs * 255).astype(np.uint8)
        labels = labels[: self.max_samples].detach().cpu().numpy().astype(np.uint8)
        preds = preds[: self.max_samples].detach().cpu().numpy().astype(np.uint8)

        out = PredData(inputs=inputs, labels=labels, preds=preds)

        return out

    
    def _save_media_to_disk(
        self,
        trainer,
        pred_data: Optional[PredData],
        mode: Mode,
        batch_idx: Optional[int] = None,
    ):
        """For a given mode (train/val/test), save the results to disk"""
        if not self.save_to_disk:
            return
        if pred_data is None:  # Empty queue
            rank_zero_warn(f"Empty queue! Mode: {mode}")
            return

        # Create output filename
        if self.save_latest_only:
            output_filename = f"results.{mode.name.lower()}.png"
        else:
            if batch_idx is None:
                output_filename = (
                    f"results-epoch{trainer.current_epoch}.{mode.name.lower()}.png"
                )
            else:
                output_filename = f"results-epoch{trainer.current_epoch}-step{batch_idx}.{mode.name.lower()}.png"
        output_filename = self.exp_dir / output_filename

        # Get the latest batches from the data queue in LightningModule
        inputs, labels, preds = pred_data.inputs, pred_data.labels, pred_data.preds

        # Colorize labels and predictions
        labels_rgb = [self.map_color_palette(lbl) for lbl in labels]
        preds_rgb = [self.map_color_palette(pred) for pred in preds]
        inputs_l = [ipt for ipt in inputs]

        # Create collage of results
        results_l = []
        # Combine each pair of inp/lbl/pred into singe image
        for inp, lbl, pred in zip(inputs_l, labels_rgb, preds_rgb):
            res_combined = np.concatenate((inp, lbl, pred), axis=1)
            results_l.append(res_combined)
        # Create grid from combined imgs
        n_imgs = len(results_l)
        n_cols = 2  # Fix num of columns
        n_rows = int(math.ceil(n_imgs / n_cols))
        img_h, img_w, _ = results_l[0].shape
        grid_results = np.zeros((img_h * n_rows, img_w * n_cols, 3), dtype=np.uint8)
        for idy in range(n_rows):
            for idx in range(n_cols):
                grid_results[
                    idy * img_h : (idy + 1) * img_h, idx * img_w : (idx + 1) * img_w, :
                ] = results_l[idx + idy]

        # Save collage
        if not cv2.imwrite(
            str(output_filename), cv2.cvtColor(grid_results, cv2.COLOR_RGB2BGR)
        ):
            rank_zero_warn(f"Error in writing image: {output_filename}")

    
    def _save_media_to_logger(self, trainer, pred_data: Optional[PredData], mode: Mode):
        """Log images to wandb at the end of a batch. Steps are common for train/val/test"""
        if not self.valid_logger:
            return
        if pred_data is None:  # Empty queue
            return

        if isinstance(trainer.logger, LoggerCollection):
            loggers = trainer.logger
        else:
            loggers = [trainer.logger]
        task_id = trainer.datamodule.task_id
        for logger in loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self._log_media_to_wandb(logger, pred_data, mode, task_id)
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                print("Still in progress to log to Tensorboard")
            elif not (isinstance(logger, LoggerCollection)):
                print(f"No method to log media to logger: {trainer.logger}")

    def _log_media_to_wandb(self, logger, pred_data: Optional[PredData], mode: Mode, task_id=0):
        # Get the latest batches from the data queue in LightningModule
        inputs, labels, preds = pred_data.inputs, pred_data.labels, pred_data.preds

        # Create wandb Image for logging
        mask_list = []
        for img, lbl, pred in zip(inputs, labels, preds):
            mask_img = wandb.Image(
                img,
                masks={
                    "predictions": {
                        "mask_data": pred,
                        "class_labels": self.datamodule.map_labels,
                    },
                    "groud_truth": {
                        "mask_data": lbl,
                        "class_labels": self.datamodule.map_labels,
                    },
                },
            )
            mask_list.append(mask_img)

        wandb_log_label = f"{mode.name.title()}/Task {task_id}/Predictions"
        if self.is_windows:
            wandb_log_label = wandb_log_label.replace("/","-")
        logger.experiment.log({wandb_log_label: mask_list}, commit=False)

