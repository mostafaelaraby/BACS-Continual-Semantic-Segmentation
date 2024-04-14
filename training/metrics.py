# adapted from https://github.com/Shreeyak/pytorch-lightning-segmentation-template/
from dataclasses import dataclass

import torch
import numpy as np
from torchmetrics import IoU as JaccardIndex
from torch import Tensor


@dataclass
class IouMetric:
    iou_per_class: Tensor
    miou: Tensor  # Mean IoU across all classes
    accuracy: Tensor
    precision: Tensor
    recall: Tensor
    specificity: Tensor


class IoU(JaccardIndex):
    def __init__(self, num_classes: int = 11, ignore_indx: int = 255):
        """Calculates the metrics iou, true positives and false positives/negatives for multi-class classification
        problems such as semantic segmentation.
        Because this is an expensive operation, we do not compute or sync the values per step.

        Forward accepts:

        - ``prediction`` (float or long tensor): ``(N, H, W)``
        - ``label`` (long tensor): ``(N, H, W)``

        Note:
            This metric produces a dataclass as output, so it can not be directly logged.
        """
        super().__init__(
            num_classes=num_classes, ignore_index=ignore_indx, reduction="none"
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Calculate the confusion matrix and accumulate it

        Args:
            prediction: Predictions of network (after argmax). Shape: [N, H, W]
            label: Ground truth. Each pixel has int value denoting class. Shape: [N, H, W]
        """
        target = target.view(-1).int()
        preds = preds.view(-1).int()
        # remove ignore index
        mask_ignored_labels = (target >= 0) & (target < self.num_classes)
        
        super().update(preds[mask_ignored_labels], target[mask_ignored_labels])

    def compute(self):
        """Compute the final IoU and other metrics across all samples seen"""
        # Calculate True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN)
        conf_mat = self.confmat
        tp = conf_mat.diagonal()
        fn = conf_mat.sum(dim=0) - tp
        fp = conf_mat.sum(dim=1) - tp
        total_px = conf_mat.sum()
        tn = total_px - (tp + fn + fp)

        # Accuracy (what proportion of predictions — both Positive and Negative — were correctly classified?)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        accuracy[torch.isnan(accuracy)] = 0
        # Precision (what proportion of predicted Positives is truly Positive?)
        precision = tp / (tp + fp)
        precision[torch.isnan(precision)] = 0

        # Recall or True Positive Rate (what proportion of actual Positives is correctly classified?)
        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0

        # Specificity or true negative rate
        specificity = tn / (tn + fp)
        specificity[torch.isnan(specificity)] = 0 
        # compute iou per class and mean
        iou_per_class = super().compute()
        mean_iou = iou_per_class.mean()
        data_r = IouMetric(
            iou_per_class=iou_per_class,
            miou=mean_iou,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
        )

        return data_r

    @staticmethod
    def get_mean_per_classes(metric_result: Tensor, classes: list):
        """computes average of a metric for a specific set of classes

        Args:
            metric_result (torch.Tensor): tensor of the metric per class
            classes (list): list of classes in the current task

        Returns:
            float: resulting average per given classes
        """
        total_miou = sum([metric_result[label].item() for label in classes])
        return total_miou / len(classes)


class PerStepResult:
    """Aggregates results computed per training CL task"""

    def __init__(self, continual):
        self._per_step_result = {
            "mIoU": [],
            "IoU-Old": [],
            "IoU-Old-nobg": [],
            "IoU-New": [],
        }
        self.metrics = self._per_step_result.keys()
        self.continual = continual
        self.task_id = 0

    def update(self, final_result):
        for metric in self.metrics:
            self._per_step_result[metric].append([])
        for dataset_id in range(len(final_result)):
            for metric in self.metrics:
                if self.continual:
                    metric_key = "test.{}/Task {}/{}".format(
                        dataset_id, self.task_id, metric
                    )
                else:
                    metric_key = "test.{}/{}".format(dataset_id, metric)
                if metric_key in final_result[dataset_id]:
                    self._per_step_result[metric][-1].append(
                        final_result[dataset_id][metric_key]
                    )
        self.task_id += 1

    def get_metrics(self):
        if not (self.continual):
            return ["mIoU"]
        current_metrics = list(self.metrics)
        current_metrics.append("Avg-IoU")
        return current_metrics

    def get_avg_iou(self):
        miou_per_task = np.array(self._per_step_result["mIoU"])
        miou_avg = miou_per_task.mean(axis=0)
        return miou_avg

    def get_n_datasets(self):
        return len(self._per_step_result["mIoU"][-1])

    def compute(self):
        results = {}
        for metric in self.metrics:
            results[metric] = self._per_step_result[metric][-1]
        results["Avg-IoU"] = self.get_avg_iou()
        return results


if __name__ == "__main__":
    # Run tests
    # Tests
    def test_iou():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create Fake label and prediction
        label = torch.zeros((1, 4, 4), dtype=torch.long, device=device)
        pred = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
        label[:, :3, :3] = 1
        pred[:, -3:, -3:] = 1
        expected_iou = torch.tensor([2.0 / 12, 4.0 / 14], device=device)

        print("Testing IoU metrics", end="")
        iou_train = IoU(num_classes=2)
        iou_train.to(device)
        iou_train(pred, label)
        metrics_r = iou_train.compute()
        iou_per_class = metrics_r.iou_per_class
        assert (iou_per_class - expected_iou).sum() < 1e-6
        print("  passed")

    print("Running tests on IoU metrics module...\n")
    test_iou()
