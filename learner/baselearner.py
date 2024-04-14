from pytorch_lightning.callbacks import Callback
from visualization import Mode
from functools import partial
import torch.nn as nn


class BaseLearner(Callback):
    def __init__(self, network, config, is_domain_shift):
        super().__init__()
        self.network = network
        self.config = config
        self.task_id = 0
        self.use_bg_detector = (
            "bg_detector" in config.training and config.training["bg_detector"]
        )
        self.is_domain_shift = is_domain_shift

    def _init_bg_detector(self):
        # initialize a head for each task to detect seen/unseen classes
        if self.task_id == 0:
            self.network.seen_fg_network = self.network.get_seen_not_seen_head()
            self.network.seen_fg_network.seen_not_seen_clf = nn.ModuleList()
        current_head = self.network.seen_fg_network.get_classification_head(1)
        self.network.seen_fg_network.seen_not_seen_clf.append(current_head)

    def on_init_start(self, trainer):
        """Called on trainer init before init of network

        Args:
            trainer (pl.Trainer): Pytorch lightning trainer
        """
        # initialize networks' classifier
        # this baselearner is used for joint training
        num_classes = self.config.dataset.dataset.num_classes
        self.network.classifier_head = self.network.get_classification_head(num_classes)
        # if transformer, do not initialize classification head. This is essential for 2 gpu training.
        if self.config.network._target_ != "networks.TranSeg":
            self.network.classifier_head = self.network.get_classification_head(
                self.config.dataset.dataset.num_classes
            )
        else:
            self.network.classifier_head = None
        if self.use_bg_detector:
            self._init_bg_detector()

    def on_fit_start(self, trainer, pl_module):
        """On fit call start

        Args:
            trainer (pl.trainer): trainer used by pytorch lightning
            pl_module (pl.lightningmodule): lightning module used
        """
        pl_module.init_metrics(trainer.datamodule.get_n_classes())
        pl_module.loss_fn.on_fit_start(
            self.task_id,
            train_dataloader=trainer.train_dataloader,
            accelerator=trainer.accelerator,
            accumulate_grad_batches=trainer.accumulate_grad_batches,
            model=pl_module.network,
        )
        # now check
        if self.is_domain_shift:
            num_classes = trainer.datamodule.num_classes
            pl_module.loss_fn.nb_new_classes = num_classes
            pl_module.loss_fn.old_classes = num_classes
            pl_module.loss_fn.nb_current_classes = num_classes

    def on_test_start(self, trainer, pl_module):
        """Called on test start

        Args:
            trainer (pl.trainer): trainer used by pytorch lightning
            pl_module (pl.lightningmodule): lightning module used
        """
        # initialize metrics used for testing
        pl_module.init_testing_metrics(
            trainer.datamodule.get_n_classes(), len(trainer.test_dataloaders)
        )

    def on_train_start(self, trainer, pl_module):
        """Called on training start

        Args:
            trainer (pl.trainer): Pytorch lightning trainer
            pl_module (pl.lightningmodule): lightning module including our network used in training
        """
        pl_module.loss_fn.on_train_start(
            self.task_id,
            trainer=trainer,
            pl_module=pl_module,
            train_dataloader=trainer.datamodule.train_dataloader(),
            datamodule=trainer.datamodule,
            accelerator=trainer.accelerator,
            accumulate_grad_batches=trainer.accumulate_grad_batches,
            model=pl_module.network,
        )

    def teardown(self, trainer, pl_module, stage=None):
        """Called on training end

        Args:
            trainer (pl.trainer): Pytorch lightning trainer
            pl_module (pl.lightningmodule): lightning module including our network used in training
        """
        # update loss function
        if stage == "fit":
            pl_module.loss_fn.on_train_end(
                task_num=self.task_id,
                model=pl_module.network,
                train_dataloader=trainer.datamodule.train_dataloader(),
                accelerator=trainer.accelerator,
                log_media=(
                    partial(pl_module.log_media.append, mode=Mode.TRAIN)
                    if pl_module.log_images
                    else None
                ),
            )
