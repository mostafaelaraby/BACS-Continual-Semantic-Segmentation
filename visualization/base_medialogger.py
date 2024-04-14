from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback
from typing import Optional
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pathlib import Path
from pytorch_lightning.loggers.base import LoggerCollection
from enum import Enum
import numpy as np
from typing import Any, List
from collections import deque
import sys


class Mode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class LogMediaQueue:
    """Holds a circular queue for each of train/val/test modes, each of which contain the latest N batches of data"""

    def __init__(self, max_len: int = 3):
        if max_len < 1:
            raise ValueError(f"Queue must be length >= 1. Given: {max_len}")

        self.max_len = max_len
        self.log_media = {
            Mode.TRAIN: deque(maxlen=self.max_len),
            Mode.VAL: deque(maxlen=self.max_len),
            Mode.TEST: deque(maxlen=self.max_len),
        }

    def clear(self):
        """Clear all queues"""
        for _, queue in self.log_media.items():
            queue.clear()

    def append(self, data: Any, mode: Mode):
        """Add a batch of data to a queue. Mode selects train/val/test queue"""
        self.log_media[mode].append(data)

    def fetch(self, mode: Mode) -> List[Any]:
        """Fetch all the batches available in a queue. Empties the selected queue"""
        data_r = []
        while len(self.log_media[mode]) > 0:
            data_r.append(self.log_media[mode].popleft())

        return data_r

    def len(self, mode: Mode) -> int:
        """Get the number of elements in a queue"""
        return len(self.log_media[mode])


def generate_colormap(N=256, normalized=False):
    """colormap to create plots of segmentation maps

    Args:
        N (int, optional): Number of classes for the colormap. Defaults to 256.
        normalized (bool, optional): to normalize colors 0-1 or keep it in 0-255 range. Defaults to False.
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class BaseMediaLogger(Callback):
    SUPPORTED_LOGGERS = [pl_loggers.WandbLogger]

    def __init__(
        self,
        datamodule,
        max_samples: int = 3,
        period_epoch: int = 1,
        period_step: int = 0,
        save_to_disk: bool = True,
        save_latest_only: bool = True,
        exp_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.datamodule = datamodule
        self.max_samples = max_samples
        self.period_epoch = period_epoch
        self.period_step = period_step
        self.save_to_disk = save_to_disk
        self.save_latest_only = save_latest_only
        try:
            self.exp_dir = Path(exp_dir) if self.save_to_disk else None
        except TypeError as e:
            raise ValueError(f"Invalid exp_dir: {exp_dir}. \n{e}")
        self.verbose = verbose
        # Project-specific fields
        self.class_colors = generate_colormap(self.datamodule.num_classes)
        # to mitigate path error when syncing images to wandb
        self.is_windows = sys.platform.startswith("win")

    def setup(self, trainer, pl_module, stage: str):
        # This callback requires a ``.log_media`` attribute in LightningModule
        req_attr = "log_media"
        if not hasattr(pl_module, req_attr):
            raise AttributeError(
                f"{pl_module.__class__.__name__}.{req_attr} not found. The {BaseMediaLogger.__name__} "
                f"callback requires the LightningModule to have the {req_attr} attribute."
            )
        if not isinstance(pl_module.log_media, LogMediaQueue):
            raise AttributeError(
                f"{pl_module.__class__.__name__}.{req_attr} must be of type {LogMediaQueue.__name__}"
            )

        if self.verbose:
            pl_module.print(
                f"Initializing Callback {BaseMediaLogger.__name__}. "
                f"Logging to disk: {self.exp_dir if self.save_to_disk else False}"
            )
        self._create_log_dir()
        self.valid_logger = True if self._logger_is_supported(trainer) else False

    def _create_log_dir(self):
        if not self.save_to_disk:
            return

        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def _logger_is_supported(self, trainer):
        """This callback only works with wandb logger"""
        for logger_type in self.SUPPORTED_LOGGERS:
            if isinstance(trainer.logger, logger_type):
                return True
            elif isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    if isinstance(logger, logger_type):
                        return True

        rank_zero_warn(
            f"Unsupported logger: '{trainer.logger}', will not log any media to logger this run."
            f" Supported loggers: {[sup_log.__name__ for sup_log in self.SUPPORTED_LOGGERS]}."
        )
        return False

    def _should_log_epoch(self, trainer):
        if trainer.sanity_checking:
            return False
        if self.period_epoch < 1 or (
            (trainer.current_epoch + 1) % self.period_epoch != 0
        ):
            return False
        return True

    def _should_log_step(self, trainer, batch_idx):
        if trainer.sanity_checking:
            return False
        if self.period_step < 1 or ((batch_idx + 1) % self.period_step != 0):
            return False
        return True

    def on_test_epoch_end(self, trainer, pl_module):
        if self._should_log_epoch(trainer):
            self._log_results(trainer, pl_module, Mode.TEST)

    def teardown(self, trainer, pl_module, stage=None):
        """Called on training end

        Args:
            trainer (pl.trainer): Pytorch lightning trainer
            pl_module (pl.lightningmodule): lightning module including our network used in training
        """
        # on training end log results of the training
        if stage == "fit":
            self._log_results(trainer, pl_module, Mode.TRAIN)

    def _log_results(
        self, trainer, pl_module, mode: Mode, batch_idx: Optional[int] = None
    ):
        raise NotImplementedError("log_results should be overriden")
