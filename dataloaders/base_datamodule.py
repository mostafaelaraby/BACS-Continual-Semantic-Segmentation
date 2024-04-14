import collections
import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import os
from continuum import SegmentationClassIncremental
from dataset import PyTorchDataset
from dataset.base_segmentation_dataset import (
    BaseBufferDataset,
    BaseMemMapDataset,
    BaseSubset,
)
from training.utils import create_iteration_sampler
import random
import torch
import numpy as np
from copy import deepcopy as copy
from torch.utils.data import Subset


class BaseDataModule(pl.LightningDataModule):
    """Parent data module for any datamodule used by pt lightning"""

    def __init__(self, train_config: DictConfig, dataset_config: DictConfig, name: str):
        """Initialize Base Data Module

        Args:
            train_conf (DictConfig): Training configuration in dict format
            dataset (DictConfig): Dataset configuration in dict format
        """
        super().__init__()
        self.seed = train_config.seed if "seed" in train_config else 1
        self.name = name
        self.batch_size = train_config.batch_size
        # continual flag depending on initial increment propery in config or not
        self.continual = (
            "initial_increment" in train_config
            and train_config.initial_increment < dataset_config.num_classes
        )
        # to use domain shift cityscape or the normal setup as in Plop
        self.is_domain_shift = (
            dataset_config.domain_shift if "domain_shift" in dataset_config else False
        )
        if train_config.num_workers == "auto":
            if hasattr(os, "sched_getaffinity"):
                try:
                    self.num_workers = len(os.sched_getaffinity(0))
                except:
                    self.num_workers = os.cpu_count()
            else:
                self.num_workers = os.cpu_count()
        else:
            self.num_workers = train_config.num_workers
        self.task_id = -1
        # check whether to validate on train or not

        self._sweep = "sweep" in train_config and train_config["sweep"]
        self.val_on_train = "val_on_train" in train_config and train_config.val_on_train
        # self.val_on_train = self.val_on_train or self._sweep
        # initialize dataset parameters
        self._init_dataset_params(dataset_config)
        # intitalize parameters related to CL learning
        self._init_continual_params(train_config)
        # shuffle classes if shuffling is enabled in CL
        self._init_shuffling(train_config)
        # debug mode to use only 3 batches per epoch
        self.debug = "debug" in train_config and train_config.debug
        # setup data directory
        self._setup_data_dir()

    def _init_dataset_params(self, dataset_config):
        """Initialize attributes related to dataset

        Args:
            dataset (dictconfig): input dataset configuration
        """
        self.crop_size = dataset_config.crop_size
        if isinstance(self.crop_size, collections.Iterable):
            self.crop_size = tuple(self.crop_size)
        self.data_dir = dataset_config.root
        self.num_classes = dataset_config.num_classes
        self.ignore_index = (
            dataset_config.ignore_index if "ignore_index" in dataset_config else 255
        )
        self.initial_increment = self.num_classes
        self.increment = 0
        self.n_tasks = 0
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_transform = None
        self.train_transform = None

    def _init_shuffling(self, train_config):
        """initialize attributes related to shuffling

        Args:
            train_config (Dict): training configuration
        """
        self.shuffle_classes = (
            train_config.shuffle_classes if "shuffle_classes" in train_config else False
        )
        self.classes_order = [i for i in range(1, self.get_n_cl_scenario() + 1)]
        self.original_class_to_ordered = {}
        self.ordered_to_original_class = {}
        if self.shuffle_classes:
            random.shuffle(self.classes_order)
            self.ordered_to_original_class = {
                i: self.classes_order[i - 1] for i in range(1, self.num_classes)
            }
        self.map_labels = {}

    def _init_continual_params(self, train_config):
        """Initialize attributes related to continual learning

        Args:
            train_config (DictConfig): Input training configuration
        """
        if self.continual:
            self.increment = train_config.increment
            self.initial_increment = train_config.initial_increment
            self.n_epochs_task = train_config.epochs
            self.mode = train_config.mode
            self.n_tasks = int(
                ((self.num_classes - self.get_initial_n_classes()) / self.increment) + 1
            )
            self.scenario = {}

    def get_initial_n_classes(self):
        """returns initial number of classes of first task or all classes in joint"""
        if self.continual:
            # add  1 for background and to avoid + 1 just override this function
            return self.initial_increment + 1
        return self.num_classes

    def get_n_classes(self):
        """returns number of classes in the current task training

        Returns:
            int: number of classes
        """
        if self.continual:
            return self.get_initial_n_classes() + self.increment * (self.task_id)
        return self.num_classes

    def get_current_task_classes(self):
        """Returns list of classes in the current task

        Returns:
            list: list of classes trained in the current task
        """
        if self.continual:
            if self.task_id > 0:
                prev_task_classes = self.get_initial_n_classes() + self.increment * (
                    self.task_id - 1
                )
                return [0] + list(range(prev_task_classes, self.get_n_classes()))
            return range(self.get_initial_n_classes())
        return range(self.num_classes)

    def transform_batch(self, batch):
        """Add extra augmentation for the labels specific to each dataset

        Args:
            batch (tuple): batch before feeding to train/test

        Returns:
            tuple: transformed batch output
        """
        return batch

    def _get_normalization_mean_std(self):
        """Override to return normalization used in order to de-normalize

        Returns:
            tuple: tuple of mean and std normalization if no normalization then return None
        """
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def _setup_data_dir(self):
        """Creates data dir used to save dataset data"""
        if not (os.path.isdir(self.data_dir)):
            os.makedirs(self.data_dir)

    def set_task_id(self, task_id):
        """Updates the datamodule task id in continual learning setup

        Args:
            task_id (int): task id of the current training step in CL
        """
        self.task_id = task_id

    def get_label_name(self, label_indx):
        """Takes class index and returns its text name for logging

        Args:
            label_indx (int): class index

        Returns:
            str: class name corresponding to input index
        """
        if self.shuffle_classes and label_indx in self.ordered_to_original_class:
            # shuffled classes by continuum are re-indexed to be zero based and names would differ
            # for that purpose we restore its ordered name
            label_indx = self.ordered_to_original_class[label_indx]
        if label_indx not in self.map_labels:
            raise ValueError("Label {} not in {} dataset".format(label_indx, self.name))
        return self.map_labels[label_indx]

    def update_continuum_datasets(self, stage=None, CustomDataset=PyTorchDataset):
        """Create continual learning specific dataset using continuum library + split dataset in case of validation

        Args:
            stage (str, optional): string including fit for training and test for val/test. Defaults to None.
        """
        # split dataset in case of val_on_train
        if self.val_on_train:
            # split the train dataset into train and val and discard the val test set
            train_len = int(0.8 * len(self.train_dataset))
            val_len = len(self.train_dataset) - train_len
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_len, val_len]
            )
        if not (self.continual):
            return
        # add property
        if stage == "fit" or stage is None:
            self.train_dataset = CustomDataset(
                self.data_dir,
                dataset_type=self.train_dataset.__class__,
                dict_args=self.train_dataset.__dict__,
                transformation=self.train_transform,
            )
            if self.val_dataset is not None:
                self.val_dataset = CustomDataset(
                    self.data_dir,
                    dataset_type=self.val_dataset.__class__,
                    dict_args=self.val_dataset.__dict__,
                    transformation=self.test_transform,
                    train=False,
                )
        if (stage == "test" or stage is None) and self.test_dataset is not None:
            self.test_dataset = CustomDataset(
                self.data_dir,
                dataset_type=self.test_dataset.__class__,
                dict_args=self.test_dataset.__dict__,
                transformation=self.test_transform,
                train=False,
            )

    def get_n_cl_scenario(self):
        """Return number of classes for continuum scenario

        Returns:
            int : number of classes
        """
        # return number of classes for continumm doesnt need bg
        return self.num_classes - 1

    def _create_scenario(
        self,
        dataset,
        with_prev_tasks=False,
        ClassIncrementalScenario=SegmentationClassIncremental,
    ):
        """Create or gets an already initialized scenario for the given dataset

        Args:
            dataset (BaseDataloader.BaseDataset): input dataset for which a scenario will be created
            with_prev_tasks (bool, optional): flag to return validation for current and prev tasks in sequential mode. Defaults to False.
        Returns:
            continuum.dataset: returns continuum dataset
        """
        dataset_id = id(dataset)
        scenario = None
        require_post_processing = False
        if with_prev_tasks and self.mode != "overlap":
            # overlap mode to include labels of prev. classes and future classes
            # as evaluated in existing literature
            scenario = ClassIncrementalScenario(
                cl_dataset=dataset,
                increment=self.increment,
                initial_increment=self.initial_increment,
                mode="overlap",
                nb_classes=self.get_n_cl_scenario(),
                transformations=dataset.transformation.transforms,
                class_order=self.classes_order,
            )
            require_post_processing = True
        if dataset_id not in self.scenario:
            self.scenario[dataset_id] = ClassIncrementalScenario(
                cl_dataset=dataset,
                increment=self.increment,
                initial_increment=self.initial_increment,
                mode=self.mode,
                nb_classes=self.get_n_cl_scenario(),
                transformations=dataset.transformation.transforms,
                class_order=self.classes_order,
            )
            require_post_processing = True
        if scenario is None:
            scenario = self.scenario[dataset_id]
        if require_post_processing:
            self._post_process_scenario(scenario)
        return scenario

    def _post_process_scenario(self, scenario):
        # add any post processing dataset specific to our scenario
        pass

    def _get_continual_dataset(self, dataset, with_prev_tasks=False):
        """creating continual scenario used to get the dataset

        Args:
            dataset (torch.dataset): base dataset used to create the scenario
            with_prev_tasks (bool, optional): flag to return validation for current and prev tasks. Defaults to False.

        Returns:
            continuum: dataset in the continuum format
        """
        # relying on object unique id to create scenarios cached for each dataset
        # train, val and test
        scenario = self._create_scenario(dataset, with_prev_tasks)
        assert self.task_id >= 0
        # return a scenario including previous steps for testing purposes
        if with_prev_tasks and self.task_id > 0:
            return scenario[: self.task_id + 1]
        return scenario[self.task_id]

    def _create_subset(self, current_dataset):
        # in debug mode each epoch will load only 1 iteration step
        # train only on 30% of the training data for the sweep
        num_iterations = int(math.ceil(0.1 * (len(current_dataset)))) if self._sweep else 6
        subset_indices = np.random.randint(
            0, len(current_dataset), size=num_iterations
        )
        subset_dataset = Subset(current_dataset, subset_indices)
        return BaseSubset(subset_dataset, use_transform=False)

    def get_dataloader(
        self,
        dataset,
        shuffle=True,
        with_prev_tasks=False,
        drop_last=False,
        num_workers=None,
        use_batch_sampler=True,
    ):
        """Returns data loader from input dataset

        Args:
            dataset (torch.dataset): input dataset that the loader will be built on
            shuffle (bool, optional): flag to shuffle the data. Defaults to True.
            with_prev_tasks (bool, optional): flag to return validation for current and prev tasks. Defaults to False.
            drop_last (bool, optional): flag to drop last not full batch or not. Defaults to False.

        Returns:
            torch.DataLoader: data loader used for the training/val/test
        """
        current_dataset = dataset
        if self.continual:
            current_dataset = self._get_continual_dataset(dataset, with_prev_tasks)
        num_workers = self.num_workers if num_workers is None else num_workers
        use_subset = self.debug or (self._sweep and use_batch_sampler)
        if use_subset:
            current_dataset = self._create_subset(current_dataset)

        return DataLoader(
            current_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else 2,
        )

    def train_dataloader(self):
        """Get training data loader

        Returns:
            torch.dataloader: train dataloader used in the current step
        """
        return self.get_dataloader(self.train_dataset, drop_last=True)

    def val_dataloader(self):
        """Get validation data loader

        Returns:
            torch.dataloader: val dataloader used in the current step
        """
        return self.get_dataloader(
            self.val_dataset, shuffle=False, use_batch_sampler=False
        )

    def test_dataloader(self):
        """Get testing data loader

        Returns:
            torch.dataloader: test dataloader used in the current step
        """
        return self.get_dataloader(
            self.test_dataset, shuffle=False, use_batch_sampler=False
        )

    def get_val_test_all(self):
        """returns a list of validation and test dataloader with all data

        Returns:
            list: list of data loaders
        """
        dataloaders = []
        dataloaders.append(
            self.get_dataloader(
                self.val_dataset,
                shuffle=False,
                with_prev_tasks=True,
                use_batch_sampler=False,
            )
        )
        if self.test_dataset is not None:
            dataloaders.append(
                self.get_dataloader(
                    self.test_dataset,
                    shuffle=False,
                    with_prev_tasks=True,
                    use_batch_sampler=False,
                )
            )
        return dataloaders

    def get_buffer_loader(self, img_dict, target_dict, target_trsf=None):
        base_dataset = []
        for key in img_dict:
            base_dataset.append((img_dict[key], target_dict[key]))
        base_dataset = np.array(base_dataset)
        dataset = BaseBufferDataset(
            base_dataset, transforms=self.train_transform, target_trsf=target_trsf
        )
        continual = copy(self.continual)
        self.continual = False
        buffer_dataloader = self.get_dataloader(
            dataset,
            shuffle=True,
            drop_last=False,
            num_workers=min(int(self.num_workers // 2), 2),
            use_batch_sampler=False, 
        )
        self.continual = continual
        return buffer_dataloader

    def get_logits_loader(
        self, imgs_map, logits_map, n_classes, length, transforms=None
    ):
        dataset = BaseMemMapDataset(
            imgs_map, logits_map, n_classes, length, transforms=transforms
        )
        continual = copy(self.continual)
        self.continual = False
        buffer_dataloader = self.get_dataloader(
            dataset,
            shuffle=True,
            drop_last=False,
            num_workers=min(int(self.num_workers // 2), 2),
            use_batch_sampler=False, 
        )
        self.continual = continual
        return buffer_dataloader
