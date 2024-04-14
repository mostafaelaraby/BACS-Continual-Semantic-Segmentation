import torch.nn as nn
import torch
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F
from .experience_replay import ExperienceReplay
from .base_loss import BaseLoss
from training.utils import freeze_network
from copy import deepcopy as copy
from pytorch_lightning.trainer.supporters import CombinedLoader
from torchvision import transforms as tvtransforms


class BACSLoss(ExperienceReplay):
    def __init__(
        self,
        name="BACS",
        ignore_index=255,
        alpha: float = 0.8,
        beta: float = 0.2,
        buffer_size: int = 50,
        replay_minibatch_size: int = 32,
        dark_plus_plus: bool = True,
        use_cosine_dist: bool = False,
        same_task: bool = False,
        ignore_rep_bg: bool = True,
        bg_weighted_ce: bool = False,
        seen_gamma: float = 2,
        seen_threshold: float = 0.5,
        seen_ukd: bool = True,
        seen_focal_alpha: float = None,
        lkd: float = 0.25,
        lkd_alpha: float = 0.2,
        lkd_threshold: float = 0.5,
        pseudo_label: bool = False,
    ):
        super().__init__(
            name,
            ignore_index=ignore_index,
            same_task=same_task,
            replay_minibatch_size=replay_minibatch_size,
            buffer_size=buffer_size,
            bg_weighted_ce=bg_weighted_ce,
        )

        # weight used for dark experience
        self.alpha = alpha
        # weight used for dark++ if enabled
        self.beta = beta
        self.dark_plus_plus = dark_plus_plus
        self.use_cosine_dist = use_cosine_dist
        self._use_der_loss = False
        self.ignore_rep_bg = ignore_rep_bg
        self.buffer = None
        self.update_buffer_every = 1
        self.bg_weighted_ce = bg_weighted_ce
        self.prev_model = None
        self.init_weighted_loss(
            gamma=seen_gamma, threshold=seen_threshold, ukd=seen_ukd
        )
        self.lkd = lkd
        self.lkd_threshold = lkd_threshold
        self.lkd_alpha = lkd_alpha
        # enabled only when explicitly set pseudo label with bg_weighted false
        self.pseudo_label = pseudo_label and not (bg_weighted_ce)
        self.seen_focal_alpha = seen_focal_alpha
        self.init_seen_focal_loss(alpha=seen_focal_alpha)

    def _init_dark_criterion(self, device):
        """Initialize dark experience criterion

        Args:
            device (torch.device): device used in training
        """
        if self.use_cosine_dist:
            self.dark_criterion = partial(
                nn.CosineEmbeddingLoss(), target=torch.ones(1, device=device)
            )
        else:
            self.dark_criterion = nn.MSELoss()

    def on_train_start(self, task_num, **kwargs):
        """An event fired when task is switched in a CL setup
        takes a set of named arguments
        """
        self.accelerator = kwargs.get("accelerator")
        self._init_dark_criterion(device=self.accelerator.root_device)
        self._init_buffer(task_num=task_num)
        if task_num > 0:
            self._use_der_loss = True
            # create our dataloader
            datamodule = kwargs.get("datamodule")
            # current buffer which is a single one
            assert self.same_task is False
            populate_buffer = self.alpha > 0 or self.beta > 0
            if populate_buffer:
                buffer = self._get_current_buffer()
                buffer_loader = datamodule.get_buffer_loader(
                    buffer.img_paths,
                    buffer.target_paths,
                    target_trsf=buffer.target_trsf,
                )
                buffer_logits_loader = datamodule.get_logits_loader(
                    buffer.dataset_map["examples"],
                    buffer.dataset_map["logits"],
                    buffer._logits_n_classes,
                    length=len(buffer.img_paths),
                    transforms=tvtransforms.Compose(
                        [
                            tvtransforms.Lambda(lambda x: torch.from_numpy(x)),
                            tvtransforms.RandomAutocontrast(p=0.5),
                        ]
                    ),
                )
                trainer = kwargs.get("trainer")
                trainer.train_dataloader = CombinedLoader(
                    {
                        "main": trainer.train_dataloader.loaders,
                        "buffer": buffer_loader,
                        "bufferlogits": buffer_logits_loader,
                    },
                    "max_size_cycle",
                )
                self.logit_transforms = tvtransforms.RandomAutocontrast(p=0.5)
        self.update_buffer_every = kwargs.get("accumulate_grad_batches", 1)
        self._iter_indx = 0
        # Do we need a pre-training step that mixes both buffer and current data to get the best possible initialization
        # in the fewest possible number of iterations
        if self.prev_model is not None:
            self.prev_model = self.prev_model.to(self.accelerator.root_device)
            freeze_network(self.prev_model)

    def on_train_end(self, **kwargs):
        """An event fired at end of training to cache the network of previous task"""
        BaseLoss.on_train_end(self, **kwargs)
        pre_last_tasks = kwargs.get("pre_last_tasks")
        if not (pre_last_tasks):
            return
        model = kwargs.get("model", None)
        train_dataloader = kwargs.get("train_dataloader", None)
        if self.buffer is None:
            self._init_buffer()
        # set prev model
        self.prev_model = model.clone()
        freeze_network(self.prev_model)
        populate_buffer = self.alpha > 0 or self.beta > 0
        if model is not None and train_dataloader is not None and populate_buffer:
            # which means we have resumed training so now we need to populate our buffer
            accelerator = kwargs.get("accelerator")
            trainer = kwargs.get("trainer")
            # move model to the right device
            model = model.to(accelerator.root_device)
            # disable shuffling
            train_dataloader.shuffle = False
            if trainer.datamodule._sweep or trainer.datamodule.debug:
                train_dataset = train_dataloader.dataset.base_dataset.dataset
                indices = train_dataloader.dataset.base_dataset.indices
                img_paths = train_dataset._x[indices]
                target_paths = train_dataset._y[indices]
                target_trsf = train_dataset.target_trsf
            else:
                train_dataset = train_dataloader.dataset
                img_paths = train_dataset._x
                target_paths = train_dataset._y
                target_trsf = train_dataset.target_trsf
            train_dataloader = accelerator.process_dataloader(train_dataloader)
            classes_weights = torch.ones(
                self.nb_current_classes, device=accelerator.root_device
            )
            classes_weights[0] = 0
            start_idx = 0
            end_idx = 0
            for batch in tqdm(train_dataloader):
                end_idx += batch[0].shape[0]
                batch = accelerator.to_device(batch)
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels, _ = batch
                labels = labels.long()
                model.enable_caching_sem_logits()
                logits = model(images)
                losses = -1 * F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.ignore_index,
                    weight=classes_weights,
                    reduction="none",
                ).view(images.shape[0], -1).mean(1)
                sem_logits = model.pop_sem_logits()
                seen_detector = self._get_seen_detector(images, model)
                self._add_to_buffer(
                    images,
                    sem_logits,
                    labels,
                    losses,
                    seen_detector=seen_detector,
                    paths=img_paths[start_idx:end_idx],
                    target_paths=target_paths[start_idx:end_idx],
                    target_trsf=copy(target_trsf),
                )
                start_idx += batch[0].shape[0]
            self.update_buffer_scores()

    def post_process_mask(self, img, mask):
        if self.pseudo_label and self.prev_model is not None:
            # updating masks for training
            pseudo_labels = self.prev_model(img).argmax(1)
            mask[mask == 0] = pseudo_labels[mask == 0]
        return mask

    def compute_loss(self, batch, model, train=True):
        """Computes loss given input image, target mask and optional predicted mask

        Args:
            img (tensor): input image
            mask (tensor): target semantic segmentation map
            model (networks.BaseNetwork): model used for training
            train (bool): flag denoting this is train or val/test

        Returns:
            tuple: loss value, and output predictions
        """
        # updating the mask with our preprocessor
        # preprocessor will be only used in upcoming tasks
        if isinstance(batch, dict):
            img = batch["main"][0]
            mask = batch["main"][1]
        else:
            img = batch[0]
            mask = batch[1]
        if train:
            mask = self.post_process_mask(img, mask)
        loss = self.compute_base_loss(
            img,
            mask,
            model,
            train=train,
            use_weighted_ce=self.bg_weighted_ce and self._use_der_loss,
            return_attentions=self._use_der_loss and train and self.lkd > 0,
        )
        if self._use_der_loss and train and self.lkd > 0:
            loss, preds_mask, old_attention, new_attention, seen_prob = loss
            loss += self._teacher_distill(old_attention, new_attention, seen_prob, mask)
        else:
            loss, preds_mask = loss
        # adding teacher distillation
        if train:
            # save logits before interpolation used in deeplab
            populate_buffer = self.alpha > 0 or self.beta > 0
            if self._use_der_loss and populate_buffer:
                loss += self._replay_der_loss(
                    model, batch["buffer"], batch["bufferlogits"]
                )
        preds_output = preds_mask.argmax(dim=1)
        return loss, preds_output

    def _teacher_distill(self, old_attention, new_attention, seen_prob, mask):
        # add a loss on tops of patches feature embedding
        # just limit updates to foreground selected by seen detector and background ;) by upsampling attention output
        def _normalize_embedding(embeddings, mask):
            feat_dim = embeddings.shape[1]
            if mask is None:
                embeddings = F.normalize(embeddings.view(-1, feat_dim), p=2, dim=-1)
                return embeddings
            # interpolate embeddings
            embeddings = F.interpolate(
                embeddings,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            embeddings = torch.where(
                mask.bool().unsqueeze(1), embeddings, torch.zeros_like(embeddings)
            )
            embeddings = torch.pow(embeddings, 2)
            # do we need to normalize ??
            return embeddings

        if self.lkd == 0:
            return 0
        mask_fg_bg = mask == 0
        if seen_prob is not None:
            # inc ase we disabled bg_weighted in our loss
            mask_fg_bg = mask_fg_bg & (seen_prob.max(1)[0] > self.lkd_threshold)
        distill_loss = (
            self.lkd
            * torch.frobenius_norm(
                _normalize_embedding(old_attention[-1], mask_fg_bg)
                - _normalize_embedding(new_attention[-1], mask_fg_bg),
                dim=-1,
            ).mean()
        )
        return distill_loss

    def _get_seen_detector(self, img, model, task_num=-1):
        seen_detector = None
        if hasattr(model, "seen_fg_network") and model.seen_fg_network is not None:
            penulimate_output = model.get_penultimate_output(img)
            seen_detector = model.seen_fg_network.get_seen_map_task(
                penulimate_output,
                self.prototypes,
                task_num,
            )

        return seen_detector

    def _add_to_buffer(
        self,
        examples,
        logits,
        labels,
        losses,
        seen_detector=None,
        paths=None,
        target_paths=None,
        target_trsf=None,
    ):
        """adds data to buffer for DER

        Args:
            examples (tensor): input images
            logits (tensor): predicted logits
            labels (list): list of unique labels per batch item used by the buffer for sampling
        """
        # what if we save the mask logits associated with the required label only
        with torch.no_grad():
            buffer = self._get_current_buffer()
            new_data = {
                "examples": examples.detach().cpu(),
                "logits": logits.detach().cpu(),
                "labels": labels.cpu(),
                "loss": losses.detach().cpu(),
                "img_paths": paths,
                "target_paths": target_paths,
                "target_trsf": [target_trsf for _ in range(len(paths))],
            }
            if seen_detector is not None:
                new_data["seen"] = seen_detector.detach().cpu()
            buffer.add_data(new_data)

    def _dark_pp(self, model, memory_data):
        if memory_data is None or not (self.dark_plus_plus):
            return 0
        (
            memory_dict,
            memory_inputs,
            _,
            memory_labels,
            _,
            _,
        ) = memory_data
        classes_weights = torch.zeros(self.nb_current_classes, device=self.device)
        start_indx = 1 if self.ignore_rep_bg else 0
        classes_weights[start_indx : self.old_classes] = 1
        if "lamdas" in memory_dict:
            lamdas = memory_dict["lamdas"]
            indices = memory_dict["indices"]
            preds_mask = model(memory_inputs.float())
            lamdas = torch.tensor(lamdas, device=preds_mask.device)
            dark_pp_loss = lamdas * F.cross_entropy(
                preds_mask,
                memory_labels.long(),
                ignore_index=self.ignore_index,
                weight=classes_weights,
                reduction="none",
            ) + (1 - lamdas) * F.cross_entropy(
                preds_mask,
                memory_labels[indices].long(),
                ignore_index=self.ignore_index,
                weight=classes_weights,
                reduction="none",
            )
            dark_pp_loss = dark_pp_loss.mean()
        else:
            dark_pp_loss, _ = self.compute_base_loss(
                memory_inputs.float(),
                memory_labels.long(),
                model,
                task_num=None,
                weights=classes_weights,
                train=True,
                use_weighted_ce=False,
            )
        return dark_pp_loss

    def _dark_logits(self, model, memory_data):
        if memory_data is None:
            return 0
        (
            _,
            memory_inputs,
            memory_logits,
            _,
            n_classes_per_logit,
            _,
        ) = memory_data
        memory_inputs = self.logit_transforms(memory_inputs)
        # add flag to return sem logits without enable caching etc...
        # that should be faster with a large margin
        sem_logits = model(memory_inputs, return_sem_logits=True)
        memory_logits = memory_logits.type_as(sem_logits)

        transplant = sem_logits.detach().clone()
        if self.same_task:
            # then we need to resize memory logits to current task size
            unique_n_classes = torch.unique(n_classes_per_logit)
            assert len(unique_n_classes) == 1
            unique_n_classes = unique_n_classes[0]
            memory_logits = torch.cat(
                [memory_logits, transplant[:, unique_n_classes:].detach().clone()],
                dim=1,
            )
        else:
            unique_n_classes, returned_indices = torch.unique(
                n_classes_per_logit, return_inverse=True
            )
            current_n_classes = sem_logits.shape[1]
            for indx, n_classes in enumerate(unique_n_classes):
                indices = returned_indices[indx]
                # Transplant logits in a good way
                if n_classes < current_n_classes:
                    memory_logits[indices, n_classes:] = (
                        transplant[indices, n_classes:].detach().clone()
                    )
        if self.ignore_rep_bg:
            memory_logits[:, 0] = sem_logits[:, 0].detach().clone()
        # do i need upsampling before applying dark knowledge???
        # is upsampling needed????
        dark_loss = self.dark_criterion(memory_logits, sem_logits)
        return dark_loss

    def _replay_der_loss(self, model, replay_batch=None, replay_logits=None):
        memory_data_logits, memory_data_labels = (None, None)
        if self.alpha != 0:
            memory_data_logits = {}
            memory_data_logits["examples"] = replay_logits[0]
            memory_data_logits["labels"] = None
            memory_data_logits = (
                memory_data_logits,
                replay_logits[0],
                replay_logits[1],
                None,
                replay_logits[2],
                None,
            )
        if self.beta != 0:
            memory_data_labels = {}
            memory_data_labels["examples"] = replay_batch[0]
            memory_data_labels["labels"] = replay_batch[1]
            memory_data_labels = (
                memory_data_labels,
                replay_batch[0],
                None,
                replay_batch[1],
                None,
                None,
            )
        # update losses of sampled buffer data
        loss = self.beta * self._dark_pp(
            model, memory_data_labels
        ) + self.alpha * self._dark_logits(model, memory_data_logits)
        return loss
