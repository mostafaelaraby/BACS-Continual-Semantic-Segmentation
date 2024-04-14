import wandb
from pytorch_lightning import loggers as pl_loggers
from .base_medialogger import BaseMediaLogger, Mode
from typing import Optional, OrderedDict
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pytorch_lightning.loggers.base import LoggerCollection
from sklearn.manifold import TSNE
import numpy as np
from dataclasses import dataclass
import torch
import pandas as pd
import plotly.express as px
import torch.nn.functional as F
from scipy.spatial import distance


@dataclass
class PredData:
    """Holds the data read and converted from the LightningModule's LogMediaQueue"""

    labels: np.ndarray
    penultimate: np.ndarray


class LogPrototypes(BaseMediaLogger):
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
        self.distance_name = "l1"

    def get_distance(self, feature1, feature2):
        """Returns distance between 2 feature vectors

        Args:
            feature1 (numpy): backbone output of first input
            feature2 (np.array): backboune output of second input

        Returns:
            float: distance between the two features
        """
        return distance.minkowski(feature1, feature2, p=1)

    @rank_zero_only
    def _get_preds_from_lightningmodule(
        self, pl_module, mode: Mode
    ) -> Optional[PredData]:
        """Fetch latest N batches from the data queue in LightningModule.
        Process the tensors as required (example, convert to numpy arrays and scale)
        """
        if pl_module.log_media.len(mode) == 0:  # Empty queue
            rank_zero_warn(
                f"\nEmpty LogMediaQueue! Mode: {mode}. Epoch: {pl_module.trainer.current_epoch}"
            )
            return None

        media_data = pl_module.log_media.fetch(mode)
        # computing penultimate layer output
        # limit the num of samples
        media_data = media_data[: self.max_samples]
        # keeps per class mean penultimate output for comparison with actual prototypes
        penultimate_outputs = []
        labels = []
        feat_dim = pl_module.network.get_penultimate_layer_dim()

        with torch.no_grad():
            for x in media_data:
                penultimate_output = pl_module.network.get_penultimate_output(
                    x["inputs"]
                )
                labels_down = F.interpolate(
                    input=x["labels"].clone().unsqueeze(dim=1).double(),
                    size=(penultimate_output.shape[2], penultimate_output.shape[3]),
                    mode="nearest",
                ).long()
                cl_present = torch.unique(input=labels_down, sorted=True)
                if cl_present[-1] == pl_module.loss_fn.ignore_index:
                    cl_present = cl_present[:-1]
                for cl in cl_present:
                    penultimate_outputs.append(
                        torch.mean(
                            penultimate_output[
                                (labels_down == cl).expand(-1, feat_dim, -1, -1)
                            ]
                            .view(penultimate_output.shape[1], -1)
                            .detach(),
                            dim=-1,
                        ).view(1, -1)
                    )
                    labels.append(cl.detach().cpu().numpy().astype(np.uint8))
        penultimate_outputs = torch.cat(penultimate_outputs, dim=0)

        # Limit the num of samples and convert to numpy
        penultimate_outputs = penultimate_outputs.detach().cpu().numpy()

        out = PredData(labels=labels, penultimate=penultimate_outputs)

        return out

    def _log_results(
        self, trainer, pl_module, mode: Mode, batch_idx: Optional[int] = None
    ):
        # Function used to log results
        preds_data = self._get_preds_from_lightningmodule(pl_module, mode)
        if preds_data is None:
            return
        repr_plot = self._plot_representations(
            trainer,
            preds_data,
            mode,
        )
        # init plot names
        plot_names = []
        fig_plots = []
        fig_plots.append(repr_plot)
        plot_names.append("tsne-representation")
        # then we can compare with pre-computed prototypes
        prototypes = pl_module.loss_fn.prototypes.detach().cpu().numpy()
        protol_dist = self._plot_proto_dist(
            trainer,
            pl_module,
            prototypes,
            preds_data,
            mode,
        )
        fig_plots.append(protol_dist)
        plot_names.append("{}-prototype".format(self.distance_name))
        # as we dont have per class prototype no need to do the cross proto distance
        # log cosine distance from background class to every other prototype class
        # cross_proto_dist = self._plot_cross_protodist(
        #     trainer, pl_module, prototypes, preds_data, mode,
        # )
        # fig_plots.append(cross_proto_dist)
        # plot_names.append("Cross-{}-Prototype".format(self.distance_name))

        for name, dist in zip(plot_names, fig_plots):
            self._save_media_to_disk(
                trainer,
                dist,
                mode,
                batch_idx,
                trainer.datamodule.task_id,
                name,
            )
            self._save_media_to_logger(
                trainer,
                name,
                dist,
                mode,
                trainer.datamodule.task_id,
            )

    def _plot_representations(self, trainer, preds_data: PredData, mode: Mode):
        # compute tsne from input preds data
        penultimate_2d = TSNE(
            n_components=2, learning_rate="auto", init="pca"
        ).fit_transform(preds_data.penultimate)
        # now scatter plotting the penultimate layer
        pen_data = {
            "x": penultimate_2d[:, 0],
            "y": penultimate_2d[:, 1],
            "Class": [
                trainer.datamodule.get_label_name(label.item())
                for label in preds_data.labels
            ],
        }
        pen_data = pd.DataFrame(pen_data)
        fig = px.scatter(
            pen_data,
            x="x",
            y="y",
            color="Class",
            color_discrete_map={
                trainer.datamodule.get_label_name(label.item()): "#%02x%02x%02x"
                % tuple(self.class_colors[label])
                for label in preds_data.labels
            },
        )
        return fig

    def _get_task_num(self, pl_module, label):
        return int(pl_module.loss_fn.label_to_task_num(np.array([label.item()]))[0])

    def _plot_proto_dist(
        self, trainer, pl_module, prototypes: np.array, preds_data: PredData, mode: Mode
    ):
        # compute cosine distances between data and prototypes

        distances = [
            self.get_distance(
                preds_data.penultimate[indx],
                prototypes[self._get_task_num(pl_module, label)],
            )
            for indx, label in enumerate(preds_data.labels)
        ]
        # now scatter plotting the penultimate layer
        pen_data = {
            "x": [
                trainer.datamodule.get_label_name(label.item())
                for label in preds_data.labels
            ],
            "y": distances,
        }
        pen_data = pd.DataFrame(pen_data)
        distances_groups = pen_data.groupby("x")
        mean_distances = pd.DataFrame(
            {
                "Class": list(distances_groups.groups.keys()),
                "mean": [
                    mean_val.item() for mean_val in distances_groups.mean().values
                ],
                "variance": [
                    var_val.item() for var_val in distances_groups.var().values
                ],
            }
        )
        fig = px.scatter(
            mean_distances,
            x="Class",
            y="mean",
            error_y="variance",
            color="Class",
            color_discrete_map={
                trainer.datamodule.get_label_name(label.item()): "#%02x%02x%02x"
                % tuple(self.class_colors[label])
                for label in preds_data.labels
            },
        )
        return fig

    def _plot_cross_protodist(
        self, trainer, pl_module, prototypes: np.array, preds_data: PredData, mode: Mode
    ):
        cross_distances = {}
        classes_list = np.unique(preds_data.labels)
        for indx, cl in enumerate(preds_data.labels):
            if cl.item() not in cross_distances:
                cross_distances[cl.item()] = []
            for other_cl in classes_list:
                if cl.item() == other_cl.item():
                    continue
                cross_distances[cl.item()].append(
                    self.get_distance(
                        preds_data.penultimate[indx],
                        prototypes[self._get_task_num(pl_module, other_cl)],
                    )
                )
        # now scatter plotting the penultimate layer
        pen_data = {
            "Class": [
                trainer.datamodule.get_label_name(label) for label in cross_distances
            ],
            "mean": [np.mean(cross_distances[label]) for label in cross_distances],
            "variance": [np.var(cross_distances[label]) for label in cross_distances],
        }
        pen_data = pd.DataFrame(pen_data)
        fig = px.scatter(
            pen_data,
            x="Class",
            y="mean",
            error_y="variance",
            color="Class",
            color_discrete_map={
                trainer.datamodule.get_label_name(label.item()): "#%02x%02x%02x"
                % tuple(self.class_colors[label])
                for label in preds_data.labels
            },
        )
        return fig

    @rank_zero_only
    def _save_media_to_disk(
        self, trainer, fig, mode: Mode, batch_idx: int, task_id: int, prefix: str
    ):
        if self.save_latest_only:
            output_filename = f"{prefix}.{mode.name.lower()}"
        else:
            if batch_idx is None:
                output_filename = (
                    f"{prefix}-epoch{trainer.current_epoch}.{mode.name.lower()}"
                )
            else:
                output_filename = f"{prefix}-epoch{trainer.current_epoch}-step{batch_idx}.{mode.name.lower()}"
        if task_id > -1:
            output_filename = f"{output_filename}-task{task_id}.png"
        else:
            output_filename += ".png"
        output_filename = str(self.exp_dir / output_filename)
        try:
            fig.write_image(output_filename, format="png", engine="kaleido")
        except:
            rank_zero_warn(f"Error in writing image: {output_filename}")

    @rank_zero_only
    def _save_media_to_logger(
        self, trainer, prefix: str, fig, mode: Mode, task_id: int
    ):
        """Log images to wandb at the end of a batch. Steps are common for train/val/test"""
        if not self.valid_logger:
            return
        if fig is None:  # Empty queue
            return

        if isinstance(trainer.logger, LoggerCollection):
            loggers = trainer.logger
        else:
            loggers = [trainer.logger]
        for logger in loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self._log_media_to_wandb(logger, prefix, fig, mode, task_id)
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                print("Still in progress to log to Tensorboard")
            elif not (isinstance(logger, LoggerCollection)):
                print(f"No method to log media to logger: {trainer.logger}")

    def _log_media_to_wandb(self, logger, prefix: str, fig, mode: Mode, task_id: int):
        if task_id > -1:
            log_key = f"{mode.name}/Task {task_id}/{prefix}"
        else:
            log_key = f"{mode.name}/{prefix}"
        if self.is_windows:
            log_key = log_key.replace("/", "-")
        logger.experiment.log({log_key: fig}, commit=False)
