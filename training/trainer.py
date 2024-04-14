from pytorch_lightning import Trainer as plTrainer
from pytorch_lightning import seed_everything
from hydra.utils import instantiate, get_original_cwd
from training.utils import get_experiment_name, garbage_collect
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import glob
from datetime import datetime
from visualization import LogPrototypes, LogMedia, LogDrift
from .model import Model
from .ood_model import OODModel
import torch
from learner import BaseLearner
from .metrics import PerStepResult
from pytorch_lightning.plugins import DDPPlugin
from typing import Optional
import torch.distributed as torch_distrib
from pytorch_lightning.utilities.distributed import rank_zero_info
from datetime import timedelta


class CustomDDP(DDPPlugin):
    def init_ddp_connection(
        self, global_rank: Optional[int] = None, world_size: Optional[int] = None
    ) -> None:
        global_rank = (
            global_rank
            if global_rank is not None
            else self.cluster_environment.global_rank()
        )
        world_size = (
            world_size
            if world_size is not None
            else self.cluster_environment.world_size()
        )
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        if not torch.distributed.is_initialized():
            torch_distrib.init_process_group(
                self.torch_distributed_backend,
                rank=global_rank,
                world_size=world_size,
                timeout=timedelta(seconds=120),
            )

            # on rank=0 let everyone know training is starting
            rank_zero_info(
                f"{'-' * 100}\n"
                f"distributed_backend={self.torch_distributed_backend}\n"
                f"All DDP processes registered. Starting ddp with {self.world_size} processes\n"
                f"{'-' * 100}\n"
            )


class Trainer:
    def __init__(self, config) -> None:
        seed_everything(config.training.seed, workers=True)
        self.config = config
        # Add crop_size and num_classes to be accessible by network through config
        if "crop_size" in self.config.network and "num_classes" in self.config.network:
            # used for transformer network only
            self.config.network.crop_size = self.config.dataset.dataset.crop_size
            self.config.network.num_classes = self.config.dataset.dataset.num_classes

        self.exp_name = (
            get_experiment_name(config)
            if "exp_name" not in config.training
            else config.training.exp_name
        )
        self.use_mixed_precision = (
            "mixed_precision" in config.training and config.training.mixed_precision
        )
        self.use_apex = self.use_mixed_precision and (
            "apex" in config.training and config.training.apex
        )
        self.n_gpus = (
            config.training.n_gpus
            if "n_gpus" in config.training
            else torch.cuda.device_count()
        )
        self.log_images = (
            "log_images" in config.training and config.training["log_images"]
        )
        self.log_prototypes = (
            "log_prototypes" in config.training and config.training["log_prototypes"]
        )
        log_drift = "log_drift" in config.training and config.training["log_drift"]
        self.log_drift = None

        self.use_bg_detector = (
            "bg_detector" in config.training and config.training["bg_detector"]
        )
        self._max_plot_samples = -1
        self._debug = "debug" in config.training and config.training["debug"]
        self.progress_bar_refresh_rate = (
            config.training["progress_bar_refresh_rate"]
            if "progress_bar_refresh_rate" in config.training
            else None
        )
        self.output_dir = os.path.join(get_original_cwd(), "output_logs", self.exp_name)
        self.data_module = instantiate(self.config.dataset, self.config.training)
        self.log_every_n_step = 50
        self._init_network()
        self._init_loss()
        self._init_pathes()
        # learner should be specified
        if "learner" not in self.config.training:
            self.learner_callback = BaseLearner(
                self.network, self.config, self.data_module.is_domain_shift
            )
        else:
            self.learner_callback = instantiate(
                self.config.training.learner,
                self.network,
                self.config,
                self.data_module.is_domain_shift,
            )
        # if ood is specified
        self.use_ood_test = "ood" in self.config
        if self.use_ood_test:
            self.ood_dataset = instantiate(self.config.ood, self.config.training)

        if log_drift:
            self.log_drift = LogDrift(
                buffer_size=10, measure_every=1 if self._debug else 4
            )
        if "allow_tf32" in config.training and config.training["allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _init_pathes(self):
        """used to initialize checkpoint pathes and to check for existing ones"""
        if not (os.path.isdir(self.output_dir)):
            os.makedirs(self.output_dir)
        ckpt_dir_path = (
            self.config.training.ckpt_dir
            if "ckpt_dir" in self.config.training
            else os.path.join(get_original_cwd(), "checkpoints")
        )
        self.ckpt_dir_path = os.path.join(
            ckpt_dir_path,
            "{}_{}-{}".format(self.exp_name, datetime.now().day, datetime.now().month),
        )
        self.resume_from_ckpt = None
        self.wandb_run_id = None
        # look for existing checkpoints
        if os.path.isdir(self.ckpt_dir_path):
            if self.data_module.continual:
                # re-evaluate and start from the first task
                self.resume_from_ckpt = []
                step_num = 0
                # avoids final ckpt saved to recover runs manually
                step_ckpt = glob.glob(
                    "{}/step_{}/[!f]*.ckpt".format(self.ckpt_dir_path, step_num),
                    recursive=True,
                )
                while len(step_ckpt) != 0:
                    self.resume_from_ckpt.append(
                        max(
                            step_ckpt,
                            key=os.path.getmtime,
                        )
                    )
                    step_num += 1
                    step_ckpt = glob.glob(
                        "{}/step_{}/[!f]*.ckpt".format(self.ckpt_dir_path, step_num),
                        recursive=True,
                    )
            else:
                self.resume_from_ckpt = max(
                    glob.glob("{}/[!f]*.ckpt".format(self.ckpt_dir_path)),
                    key=os.path.getmtime,
                )
            # No need to continue on same run id uncomment to enable
            # self.wandb_run_id = re.search(
            #     r"___([^ ]+)___", os.path.split(self.resume_from_ckpt)[-1], re.I
            # )[1]

    def _init_callbacks(self):
        """initialize callbacks used in pytorch lightning"""
        self.callbacks = []
        ckpt_dir_path = self.ckpt_dir_path
        if self.data_module.continual:
            task_id = self.data_module.task_id
            ckpt_dir_path = os.path.join(ckpt_dir_path, "step_{}".format(task_id))
        self.data_module.prepare_data()
        self.data_module.setup()
        # save checkpoint twice within an epoch
        self.log_every_n_step = len(self.data_module.train_dataset) // (
            (2 * self.data_module.batch_size * self.n_gpus) + 1
        )
        model_checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir_path,
            filename="{}___{}___".format(
                "{epoch:02d}_{val_loss:.2f}", self.loggers[0].version
            ),
            save_last=True,
            every_n_val_epochs=1,
        )
        self.callbacks.append(model_checkpoint)
        self.callbacks.append(self.learner_callback)
        if self.log_images:
            self._max_plot_samples = 3 if self._debug else 6
            media_logging = LogMedia(
                self.data_module,
                exp_dir=self.output_dir,
                verbose=False,
                max_samples=self._max_plot_samples,
            )
            self.callbacks.append(media_logging)
        if self.log_prototypes:
            self._max_plot_samples = 5 if self._debug else 100
            proto_logging = LogPrototypes(
                self.data_module,
                exp_dir=self.output_dir,
                verbose=False,
                max_samples=self._max_plot_samples,
            )
            self.callbacks.append(proto_logging)
        if self.log_drift is not None:
            self.callbacks.append(self.log_drift)

    def _init_loggers(self):
        """initialize loggers"""
        self.loggers = []
        wandb.login()
        wandb_logger = WandbLogger(
            name=self.exp_name,
            save_dir=self.output_dir,
            id=self.wandb_run_id,
        )
        wandb_logger.log_hyperparams(self.config)
        # first logger is always wandb to get its run id
        self.loggers.append(wandb_logger)

    def _init_network(self):
        """Initialize network"""
        self.network = instantiate(self.config.network)

    def _init_loss(self):
        """Initialize loss function used"""
        self.loss_fn = instantiate(
            self.config.loss, ignore_index=self.data_module.ignore_index
        )
        if self.log_prototypes or self.use_bg_detector:
            # then we need to init prototypes loss on top of selected top
            self.loss_fn.init_prototype_compute()
        self.loss_fn.set_continual_task_size(
            self.data_module.get_initial_n_classes(), self.data_module.increment
        )

    def _get_resume_ckpt(self):
        """Get the checkpoint to resume from in current task

        Returns:
            str: path of the checkpoint to resume from
        """
        resume_ckpt = None
        if self.resume_from_ckpt is not None and not (self._debug):
            if self.data_module.continual and self.learner_callback.task_id < len(
                self.resume_from_ckpt
            ):
                resume_ckpt = self.resume_from_ckpt[self.learner_callback.task_id]
            elif not (self.data_module.continual):
                resume_ckpt = self.resume_from_ckpt
        return resume_ckpt

    def _get_pl_module(self, task_id=-1):
        """Create pytorch lightning model

        Args:
            task_id (int, optional): step number in continual learning. Defaults to -1.

        Returns:
            pl.lightningmodule: lightning module used for training
        """
        model_cls = Model
        if self.use_bg_detector:
            model_cls = OODModel
        pl_model = model_cls(
            self.network,
            self.config.optimizer,
            self.config.training,
            self.loss_fn,
            self.config.scheduler if "scheduler" in self.config else None,
            task_id,
        )
        pl_model._init_log_media(self._max_plot_samples)
        return pl_model

    def _create_pl_trainer(self):
        """Create pytorch lightning trainer

        Returns:
            pl.Trainer: trainer used for pytorch lightning
        """
        self._init_loggers()
        self._init_callbacks()

        # Log Class order with Wandb
        self.loggers[0].log_hyperparams(
            {"classes_order": self.data_module.classes_order}
        )
        print("Current Class Order:" + str(self.data_module.classes_order) + "\n")

        accumulate_gradients = (
            self.config.training.accumulate_gradients
            if "accumulate_gradients" in self.config.training
            else 1
        )
        strategy = None
        plugins = []
        if self.n_gpus > 1 and torch.cuda.is_available():
            strategy = CustomDDP(find_unused_parameters=self.data_module.task_id > 0)
            plugins.append(strategy)
        max_steps = None
        epochs = self.config.training.epochs
        if "next_epochs" in self.config.training and self.data_module.task_id > 0:
            epochs = self.config.training.next_epochs
        if "steps_per_class" in self.config.training:
            epochs = None
            max_steps = (
                self.config.training.steps_per_class
                * self.data_module.get_n_new_task_classes()
            )

        return plTrainer(
            resume_from_checkpoint=self._get_resume_ckpt(),
            logger=self.loggers,
            callbacks=self.callbacks,
            max_epochs=epochs,
            max_steps=max_steps,
            accumulate_grad_batches=accumulate_gradients,
            check_val_every_n_epoch=self.config.training.val_every,
            log_every_n_steps=self.log_every_n_step,
            precision=16 if self.use_mixed_precision else 32,
            amp_level="O1" if self.use_apex else None,
            amp_backend="apex" if self.use_apex else "native",
            num_sanity_val_steps=False,
            gpus=self.n_gpus if torch.cuda.is_available() else None,
            plugins=plugins,
            accelerator=(
                "ddp" if self.n_gpus > 1 and torch.cuda.is_available() else None
            ),
            gradient_clip_val=2.0,
            gradient_clip_algorithm="value",
            progress_bar_refresh_rate=self.progress_bar_refresh_rate,
        )

    def _log_final_results(self, per_step_metric):
        """Logs results aggregated across different tasks

        Args:
            per_step_metric (PerStepMetric): metric including values of results of all previous steps
        """
        per_step_results = per_step_metric.compute()
        current_metrics = per_step_metric.get_metrics()
        for dataset_id in range(per_step_metric.get_n_datasets()):
            for metric in current_metrics:
                if len(per_step_results[metric]) > dataset_id:
                    self.loggers[0].log_metrics(
                        {
                            "Final/test.{}/{}".format(
                                dataset_id, metric
                            ): per_step_results[metric][dataset_id]
                        }
                    )

    def _run_test(self, trainer, pl_model):
        """Testing step

        Args:
            trainer (pl.Trainer): trainer used to run pytorch lightning tests
            pl_model (pl.lightningmodyle): pytorch lightning moduyle

        Returns:
            dict: final metrics measured on test set
        """
        test_data_loaders = self.data_module.get_val_test_all()
        final_test_metrics = trainer.test(pl_model, test_dataloaders=test_data_loaders)
        return final_test_metrics

    def _run_step(self, task_id=-1):
        """Run a training task

        Args:
            task_id (int, optional): task id used in training CL, -1 for joint training. Defaults to -1.

        Returns:
            dict: dictionary of final result
        """
        trainer = self._create_pl_trainer()
        pl_model = self._get_pl_module(task_id)
        trainer.fit(
            pl_model,
            datamodule=self.data_module,
        )
        # save checkpoint
        ckpt_dir_path = self.ckpt_dir_path
        if self.data_module.continual:
            ckpt_dir_path = os.path.join(ckpt_dir_path, "step_{}".format(task_id))
        trainer.save_checkpoint(os.path.join(ckpt_dir_path, "final.ckpt"))
        final_test_metrics = self._run_test(trainer, pl_model)
        if self.use_ood_test:
            self.ood_dataset.continual = False
            pl_model.test_ood = True
            self.ood_dataset.prepare_data()
            self.ood_dataset.setup()
            trainer.test(pl_model, test_dataloaders=self.ood_dataset.get_val_test_all())
        del trainer
        return final_test_metrics

    def fit(self):
        """Run Training"""
        per_step_metric = PerStepResult(self.data_module.continual)
        if self.data_module.continual:
            n_tasks = self.data_module.n_tasks
            # save miou per task to get average performance
            for task_id in range(n_tasks):
                self.loss_fn.last_task = task_id == (n_tasks - 1)
                self.data_module.set_task_id(task_id)
                final_test_metrics = self._run_step(task_id)
                per_step_metric.update(final_test_metrics)
                garbage_collect()
            self._log_final_results(per_step_metric)
        else:
            final_test_metrics = self._run_step()
            per_step_metric.update(final_test_metrics)
            self._log_final_results(per_step_metric)
        per_step_results = per_step_metric.compute()
        return per_step_results["mIoU"][-1]
