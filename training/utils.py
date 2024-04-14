import torch
from .loss_utils import entropy
from torch.utils.data import SequentialSampler, BatchSampler
from pytorch_lightning.utilities.memory import garbage_collection_cuda
import time
import gc
from torchvision.datasets.utils import download_url
import os
import tarfile


def adjust_learning_rate(optimizer, new_learning_rate):
    """Adjusts learning rate of the optimizer with a new learning rate

    Args:
        optimizer (nn.optim): optimizer
        new_learning_rate (float): new learning rate for the optimizer
    """
    for param in optimizer.param_groups:
        param["lr"] = new_learning_rate


# Garbage collector  https://github.com/PyTorchLightning/pytorch-lightning/issues/8430
def garbage_collect():
    if torch.cuda.is_available():
        garbage_collection_cuda()
        time.sleep(5)
        torch.cuda.empty_cache()
        garbage_collection_cuda()
    gc.collect()


def freeze_network(model):
    for par in model.parameters():
        par.requires_grad = False
    model.eval()


# Extracts dataset related statistics used in PLOP for thresholding the confidence
def find_median(
    train_loader,
    nb_current_classes,
    model_old,
    device,
    to_device,
    mode="entropy",
    threshold=0.001,
    step_threshold=None,
    step_number=None,
):
    """Find the median prediction score per class with the old model.

    Computing the median naively uses a lot of memory, to allievate it, instead
    we put the prediction scores into a histogram bins and approximate the median.

    https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram

    Args:
        train_loader (torch.dataloader): data loader used to compute median entropy/ probability
        nb_current_classes (int): number of classes in current task
        model_old (networks.BaseNetwork): model trained on previous task
        device (torch.device): device used during training can be gupu/cpu/tpu
        mode (str, optional):median computation mode can be entropy/probability. Defaults to "entropy".
        threshold (float, optional): base threshold to be used in case median is worst. Defaults to 0.001.
        step_threshold (float, optional): adapting base threshold on step number. Defaults to None.
        step_number (int, optional): step number used when step_threshold is set. Defaults to None.

    Returns:
        tuple: tuple of thresholds and max value
    """
    if mode == "entropy":
        max_value = torch.log(torch.tensor(nb_current_classes, device=device).float())
        nb_bins = 100
    else:
        max_value = 1.0
        nb_bins = 20  # Bins of 0.05 on a range [0, 1]

    histograms = torch.zeros(nb_current_classes, nb_bins).long().to(device)

    for batch in train_loader:
        batch = to_device(batch)
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels, _ = batch
        labels = labels.long()

        outputs_old = model_old(images, return_attentions=False)

        mask_bg = labels == 0
        probas = torch.softmax(outputs_old, dim=1)
        max_probas, pseudo_labels = probas.max(dim=1)

        if mode == "entropy":
            values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
        else:
            values_to_bins = max_probas[mask_bg].view(-1)

        x_coords = pseudo_labels[mask_bg].view(-1)
        y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

        histograms.index_put_(
            (x_coords, y_coords),
            torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
            accumulate=True,
        )

    thresholds = torch.zeros(
        nb_current_classes, dtype=torch.float32, device=device
    )  # zeros or ones? If old_model never predict a class it may be important

    for c in range(nb_current_classes):
        total = histograms[c].sum()
        if total <= 0.0:
            continue

        half = total / 2
        running_sum = 0.0
        for lower_border in range(nb_bins):
            lower_border = lower_border / nb_bins
            bin_index = int(lower_border * nb_bins)
            if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                break
            running_sum += lower_border * nb_bins

        median = lower_border + (
            (half - running_sum) / histograms[c, bin_index].sum()
        ) * (1 / nb_bins)

        thresholds[c] = median

    base_threshold = threshold
    if "_" in mode:
        mode, base_threshold = mode.split("_")
        base_threshold = float(base_threshold)
    if step_threshold is not None:
        threshold += step_number * step_threshold

    if mode == "entropy":
        for c in range(len(thresholds)):
            thresholds[c] = max(thresholds[c], base_threshold)
    else:
        for c in range(len(thresholds)):
            thresholds[c] = min(thresholds[c], base_threshold)
    return thresholds.to(device), max_value


class IterationBasedBatchSampler:
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def create_iteration_sampler(current_dataset, batch_size, num_iterations=1):
    batch_sampler = BatchSampler(
        SequentialSampler(range(len(current_dataset))),
        batch_size=batch_size,
        drop_last=True,
    )
    batch_sampler2 = IterationBasedBatchSampler(
        batch_sampler, num_iterations=num_iterations
    )
    return batch_sampler2


def get_experiment_name(config):
    """returns wandb experiment name

    Args:
        config (dict): configuration dictionary

    Returns:
        str: wandb experiment name
    """
    continual_info = "joint"
    if "initial_increment" in config.training:
        continual_info = "cont_{}_{}".format(
            config.training.initial_increment, config.training.increment
        )
    return "{}_{}_{}_epoch{}_batch{}_{}".format(
        config.training.name,
        config.loss.name,
        config.optimizer["_target_"].split(".")[-1],
        config.training.epochs,
        config.training.batch_size,
        continual_info,
    ).replace(" ", "")


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    if "zip" in filename:
        import zipfile

        with zipfile.ZipFile(os.path.join(root, filename), "r") as zip_ref:
            zip_ref.extractall(root)
    else:
        with tarfile.open(os.path.join(root, filename), "r") as tar:
            tar.extractall(path=root)


class TransformLabel:
    """Transform input labels using a dictionary.

    :param input_dict: (dict): Desired Dictionary transformation`
    :param masking_value: (int, optional): default value if label not present in dict
    """

    def __init__(
        self, input_dict, masking_value, inverted_order=None, inverted_masking=None
    ):
        self.input_dict = input_dict
        self.masking_value = masking_value
        self.inverted_order = inverted_order
        self.inverted_masking = inverted_masking

    def _update_lbl(self, lbl, input_dict, mask_val):
        labels_in_input = lbl.unique()
        for current_lbl in labels_in_input:
            current_lbl = int(current_lbl.item())
            if current_lbl in input_dict:
                lbl[lbl == current_lbl] = input_dict[current_lbl]
            else:
                lbl[lbl == current_lbl] = mask_val
        return lbl

    def __call__(self, lbl):
        """
        :param lbl: (Tensor): Label to be transformed.
        :return: Label Tensor: transformed label.
        """
        lbl = self._update_lbl(lbl, self.input_dict, self.masking_value)
        if self.inverted_order is not None:
            lbl = self._update_lbl(lbl, self.inverted_order, self.inverted_masking)
        return lbl

    def __repr__(self):
        return self.__class__.__name__