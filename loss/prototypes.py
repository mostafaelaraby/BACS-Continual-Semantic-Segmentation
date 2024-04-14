# prototype computation borrowed from https://github.com/LTTM/SDR
from random import randint
from .base_loss import BaseLoss
import torch.nn.functional as F
import torch
from tqdm import tqdm


class Prototypes(BaseLoss):
    """Basic CrossEntropy loss used for finetuning baseline"""

    def __init__(
        self,
        name="Prototypes",
        ignore_index=255,
    ):
        """initialize a new loss

        Args:
            name (str): name of the loss function
            ignore_index (int): index of the ignore pixel
        """
        super().__init__(name, ignore_index)
        self._prototypes_tensors = None
        self._count_features = None

    @property
    def prototypes(self):
        return self._prototypes_tensors

    def are_prototypes_ready(self):
        """Detects if at least a data point per class seen so far

        Returns:
            bool: flag to denote if all our prototypes are nonzeros
        """
        return (
            self._count_features is not None
            and self._count_features.count_nonzero() == self._count_features.shape[0]
        )

    def on_train_start(self, task_num, **kwargs):
        """An event fired when task is switched in a CL setup
        takes a set of named arguments
        """
        # initialize prototypes for the current upcoming task
        model = kwargs.get("model")
        penultimate_dim = model.get_penultimate_layer_dim()
        # we will need to send the model itself
        accelerator = kwargs.get("accelerator")
        self._init_prototypes(task_num, accelerator, penultimate_dim)

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
                [1, penultimate_dim],
                device=accelerator.root_device,
                requires_grad=False,
            )
            new_count_features = torch.zeros(
                [1], device=accelerator.root_device, requires_grad=False
            )
            self._prototypes_tensors = torch.cat(
                [self._prototypes_tensors, new_prototypes], dim=0
            )
            self._count_features = torch.cat(
                [self._count_features, new_count_features], dim=0
            )
        else:
            self._prototypes_tensors = torch.zeros(
                [1, penultimate_dim],
                device=accelerator.root_device,
                requires_grad=False,
            )
            self._count_features = torch.zeros(
                [1],
                dtype=torch.long,
                device=accelerator.root_device,
                requires_grad=False,
            )
        self._prototypes_tensors.requires_grad = False
        self._count_features.requires_grad = False

    def on_train_end(self, **kwargs):
        """An event fired at end of training to cache the network of previous task"""
        model = kwargs.get("model", None)
        train_dataloader = kwargs.get("train_dataloader", None)
        accelerator = kwargs.get("accelerator", None)
        if model is None or train_dataloader is None or accelerator is None:
            return
        # if we have any feature from any class with count zero we need to loop over train data loader
        if not (self.are_prototypes_ready()):
            # which means we have resumed training so now we need to populate our buffer
            append_media = kwargs.get("log_media")
            model = model.to(accelerator.root_device)
            train_dataloader = accelerator.process_dataloader(train_dataloader)
            for batch in tqdm(train_dataloader):
                batch = accelerator.to_device(batch)
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels, _ = batch
                images = images
                labels = labels.long()
                self.update_prototypes(
                    model,
                    images,
                    labels,
                )
                # we need to add to the queue
                if append_media is not None:
                    append_media(
                        {
                            "inputs": images,
                            "labels": labels,
                        }
                    )

    def update_feats_prototypes(self, features, target, labels_down=None):
        """Updates prototypes using penultimate layer output

        Args:
            features (Tensor): penultimate layer output
            target (Tensor): targets for the output features

        Returns:
            None: return None if update failed
        """
        features = features.detach()
        extracted_indx_proto = self._extract_labels_prototype_index(
            features, target, labels_down=labels_down
        )
        if extracted_indx_proto is None:
            return None
        (
            features,
            feat_dim,
            labels_down,
            task_nums,
        ) = extracted_indx_proto
        for task_num in task_nums:
            masked_cl = sum(labels_down == i for i in task_nums[task_num]).bool()
            n_features = masked_cl.sum()
            if n_features == 0:
                continue
            features_cl = features[masked_cl.expand(-1, feat_dim, -1, -1)].view(
                features.shape[1], -1
            )
            features_cl_sum = torch.sum(features_cl, dim=-1)
            features_running_mean_tot_cl = (
                features_cl_sum
                + self._count_features[task_num] * self._prototypes_tensors[task_num]
            ) / (self._count_features[task_num] + n_features)
            self._count_features[task_num] += n_features
            self._prototypes_tensors[task_num] = features_running_mean_tot_cl

    def update_prototypes(self, model, img, target):
        """Updates the prototypes based on current batchd data

        Args:
            model (networks.basemodel): input training model
            img (nn.Tensor): tensor of images in current batch
            target (nn.Tensor): tensor of corresponding targets
            current_weight (int): weight given to current task
        """
        features = model.get_penultimate_output(img)
        self.update_feats_prototypes(features, target)

    def _extract_labels_prototype_index(
        self, features, target, include_bg=False, labels_down=None
    ):
        feat_dim = features.shape[1]
        if labels_down is None:
            labels_down = F.interpolate(
                input=target.clone().unsqueeze(dim=1).double(),
                size=(features.shape[2], features.shape[3]),
                mode="nearest",
            ).long()
        # now we interpolate feature space to our mask size
        cl_present = torch.unique(input=labels_down, sorted=True)
        if cl_present[-1] == self.ignore_index:
            cl_present = cl_present[:-1]
        masked_cl = (labels_down != 0) & (labels_down != self.ignore_index)
        n_features = masked_cl.sum()
        if n_features == 0:
            return None
        # now we need to group cl_present by task_number
        task_nums = {}
        current_tasks = self.label_to_task_num(cl_present)
        for cl_indx, cl in enumerate(cl_present):
            if cl == 0 and not (include_bg):
                continue
            current_task = int(current_tasks[cl_indx])
            if current_task not in task_nums:
                task_nums[current_task] = []
            task_nums[current_task].append(cl)
        return features, feat_dim, labels_down, task_nums

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            batch (tuple): a tuple of image and mask
            model (networks.BaseNetwork): model used for training
            train (bool): flag denoting this is train or val/test
        """
        if isinstance(batch, dict):
            img = batch["main"][0]
            mask = batch["main"][1]
        else:
            img = batch[0]
            mask = batch[1]
        if train:
            # update prototypes in case of training
            self.update_prototypes(model, img, mask)
        else:
            # use prototypes to update the mask to include previous classes
            pass
        loss, preds_mask = self.compute_base_loss(img, mask, model, train=train)
        preds_output = preds_mask.argmax(dim=1)
        return loss, preds_output
