from .base_loss import BaseLoss
from training.buffer import Buffer
import torch
from tqdm import tqdm
from training.loss_utils import *
from torch.utils.data import SequentialSampler, BatchSampler
import numpy as np
from scipy.special import softmax


class ExperienceReplay(BaseLoss):
    def __init__(
        self,
        name="Experience Replay",
        ignore_index=255,
        alpha: float = 1.0,
        buffer_size: int = 50,
        replay_minibatch_size: int = 32,
        bg_weighted_ce: bool = False,
        same_task: bool = True,
    ):
        super().__init__(name, ignore_index=ignore_index)
        self.buffer_size = buffer_size
        self.replay_minibatch_size = replay_minibatch_size
        self.alpha = alpha
        self._use_er_loss = False
        self.buffer = None
        self.bg_weighted_ce = bg_weighted_ce
        self.same_task = same_task
        self._iter_indx = 0
        self.co_occurence_map = None
        self.grads = {}

    def _init_buffer(self, task_num=0):
        """initialize loss"""
        if task_num == 0:
            if self.same_task:
                self.buffer = [
                    Buffer(
                        self.buffer_size,
                        "task_{}".format(
                            0,
                        ),
                    )
                ]
            else:
                self.buffer = Buffer(
                    self.buffer_size,
                    "all_tasks",
                )
        elif self.same_task:
            self.buffer.append(
                Buffer(self.buffer_size, "task_{}".format(task_num), task_num=task_num)
            )
        buffer = self._get_current_buffer()
        buffer.update_task(task_num=task_num, new_class_size=self.nb_current_classes)

    def get_available_tasks(self):
        if self.buffer is None:
            return None
        if self.same_task:
            return range(len(self.buffer))
        return self.buffer.get_available_tasks()

    def _get_current_buffer(self):
        if self.same_task:
            return self.buffer[-1]
        return self.buffer

    def update_buffer_scores(self):
        if self.same_task:
            for buffer in self.buffer:
                buffer.merge_scores(self.co_occurence_map)
        else:
            self._get_current_buffer().merge_scores(self.co_occurence_map)

    def _get_random_buffer(self):
        if self.same_task:
            # randomization doesnt include current task
            scores = 1.0
            n_buffers = len(self.buffer[:-1])
            if n_buffers > 1:
                scores = np.array(
                    [self.buffer[i].get_importance() for i in range(n_buffers)]
                )
                # normalize
                scores = scores / np.max(scores)
                scores = softmax(scores)
            task_id = (
                np.random.choice(range(n_buffers), p=scores, size=1)[0]
                if n_buffers > 1
                else 0
            )
            buffer = self.buffer[task_id]
        else:
            buffer = self._get_current_buffer()
        return buffer

    def on_train_batch_start(self, **kwargs):
        BaseLoss.on_train_batch_start(self, **kwargs)
        self._iter_indx = kwargs.get("batch_idx")

    def on_train_start(self, task_num, **kwargs):
        """An event fired when task is switched in a CL setup
        takes a set of named arguments
        """
        self._iter_indx = 0
        self._init_buffer(task_num=task_num)
        if task_num > 0:
            self._use_er_loss = True

    def on_train_end(self, **kwargs):
        """An event fired at end of training to cache the network of previous task"""
        super().on_train_end(**kwargs)
        pre_last_tasks = kwargs.get("pre_last_tasks")
        if not (pre_last_tasks):
            return
        model = kwargs.get("model", None)
        train_dataloader = kwargs.get("train_dataloader", None)
        if self.buffer is None:
            self._init_buffer()
        if model is not None and train_dataloader is not None:
            # which means we have resumed training so now we need to populate our buffer
            accelerator = kwargs.get("accelerator")
            # move model to the right device
            model = model.to(accelerator.root_device)
            train_dataloader = accelerator.process_dataloader(train_dataloader)
            classes_weights = torch.ones(
                self.nb_current_classes, device=accelerator.root_device
            )
            classes_weights[0] = 0
            for index, batch in tqdm(enumerate(train_dataloader)):
                batch = accelerator.to_device(batch)
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels, _ = batch
                images = images
                labels = labels.long()
                logits = model(images)
                losses = -1 * F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.ignore_index,
                    weight=classes_weights,
                    reduction="none",
                ).view(images.shape[0], -1).mean(1)
                self._add_to_buffer(images, labels, losses)
                if (index * images.shape[0]) >= (self.buffer_size):
                    break
            self.update_buffer_scores()

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            batch (tuple): tupe of image and mask
            model (networks.BaseNetwork): model used for training
            train (bool): flag denoting this is train or val/test

        Raises:
            NotImplementedError: needs to be overriden in child classes
        """
        if isinstance(batch, dict):
            img = batch["main"][0]
            mask = batch["main"][1]
        else:
            img = batch[0]
            mask = batch[1]
        super(ExperienceReplay, self).compute_loss(img, mask, model)
        loss, preds_mask = self.compute_base_loss(
            img,
            mask,
            model,
            train=train,
            use_weighted_ce=self.bg_weighted_ce and self._use_er_loss and train,
        )
        if train:
            buffer = self._get_random_buffer()
            if self._use_er_loss:
                loss += self.alpha * self._replay_er_loss(model, buffer)
        preds_output = preds_mask.argmax(dim=1)
        return loss, preds_output

    def _add_to_buffer(self, examples, labels, losses):
        """adds data to buffer for ER

        Args:
            examples (tensor): input images
            labels (list): list of unique labels per batch item used by the buffer for sampling
        """
        # what if we save the mask logits associated with the required label only
        with torch.no_grad():
            buffer = self._get_current_buffer()
            new_data = {
                "examples": examples.detach().cpu(),
                "labels": labels.cpu(),
                "loss": losses.detach().cpu(),
            }
            buffer.add_data(new_data)

    def _update_prototype(self, feats, labels):
        # updating prototypes of experience replay in a slow manner to avoid overfitting
        if self._prototypes is not None and self.same_task:
            self._prototypes.update_feats_prototypes(feats, labels)

    def _sample_buffer(
        self, buffer=None, same_task=False, task_num=None, mixup=False, on_cpu=False
    ):
        if self.same_task and task_num is not None and task_num < len(self.buffer):
            buffer = self.buffer[task_num]
        elif buffer is None:
            buffer = self._get_random_buffer()
        if not (buffer.is_empty()):
            memory_dict = buffer.get_data(
                self.replay_minibatch_size,
                same_task=same_task,
                task_num=task_num,
                mixup=mixup,
                device=self.accelerator.root_device,
            )
            memory_inputs = memory_dict["examples"]
            memory_logits = memory_dict["logits"]
            memory_labels = memory_dict["labels"]
            # upsample logits
            if not (on_cpu):
                (
                    memory_inputs,
                    memory_logits,
                    memory_labels,
                ) = self.accelerator.to_device(
                    (memory_inputs, memory_logits, memory_labels)
                )
            return (
                memory_dict,
                memory_inputs,
                memory_logits,
                memory_labels,
                memory_dict["n_classes"],
                buffer.task_num if not (same_task) else memory_dict["task_id"],
            )
        return None

    def _replay_er_loss(self, model, buffer):
        memory_data = self._sample_buffer(buffer)
        if memory_data is None or not (self._use_er_loss):
            return 0
        (
            _,
            memory_inputs,
            _,
            memory_labels,
            _,
            task_num,
        ) = memory_data
        classes_weights = torch.zeros(self.nb_current_classes, device=self.device)
        if task_num > -1:
            old_classes = self.get_n_old_classes(task_num + 1)
            classes_weights[1:old_classes] = 1
        else:
            classes_weights[1 : self.old_classes] = 1
        experience_replay_loss, _ = self.compute_base_loss(
            memory_inputs,
            memory_labels,
            model,
            weights=classes_weights,
            task_num=task_num,
            train=True,
            use_weighted_ce=False,
        )
        experience_replay_loss *= self.alpha
        return experience_replay_loss
