from torchvision.datasets import Cityscapes
from dataset.cityscape_dataset import map_labels
from dataset.cityscape_domain_dataset import map_labels_domain
from dataset import CityscapeDomainDataset, CityScapeDomainScenario, CityScapeScenario
from dataset.continuum_dataset import PyTorchDataset
from .base_datamodule import BaseDataModule, create_iteration_sampler
from dataset import BaseSegmentationDataset
from continuum.transforms import segmentation as transforms
from omegaconf import DictConfig
import os
import requests
import numpy as np
from torch.utils.data import DataLoader
import random
from continuum import SegmentationClassIncremental
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy


def subset(dataset, indices):
    dataset._x = dataset._x[indices]
    dataset._y = dataset._y[indices]
    dataset._t = dataset._t[indices]
    return dataset


class CityscapeDataModule(BaseDataModule):
    """Cityscape datamodule for pt lightning"""

    def __init__(self, train_conf: DictConfig, dataset: DictConfig):
        # cross domain training
        self.n_cities = 21
        super(CityscapeDataModule, self).__init__(train_conf, dataset, name="CityScape")
        mean, std = self._get_normalization_mean_std()
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.crop_size, (0.5, 2.0)),
                transforms.RandomHorizontalFvlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(self.crop_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        # username and password used to download cityscape dataset
        self.username = dataset.username
        self.password = dataset.password
        # map index to label
        self.map_labels = map_labels_domain if self.is_domain_shift else map_labels

    def _init_continual_params(self, train_config):
        """Initialize attributes related to continual learning

        Args:
            train_config (DictConfig): Input training configuration
        """
        if not (self.is_domain_shift):
            return super()._init_continual_params(train_config)
        if self.continual:
            self.increment = train_config.increment
            self.initial_increment = train_config.initial_increment
            self.n_epochs_task = train_config.epochs
            self.mode = train_config.mode
            if self.continual:
                self.n_tasks = int(
                    ((self.n_cities - self.initial_increment) / self.increment) + 1
                )
            self.scenario = {}
            self._tmp_train = None
            self._tmp_val = None

    def _init_shuffling(self, train_config):
        """initialize attributes related to shuffling

        Args:
            train_config (Dict): training configuration
        """
        if not (self.is_domain_shift):
            return super()._init_shuffling(train_config)
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

    def get_n_classes(self):
        """returns number of classes in the current task training

        Returns:
            int: number of classes
        """
        if not (self.is_domain_shift):
            return super().get_n_classes()
        return self.num_classes

    def get_n_cl_scenario(self):
        """returns number of classes for continuum scenarios

        Returns:
            int: number of classes
        """
        if not (self.is_domain_shift):
            return super().get_n_cl_scenario()
        return self.n_cities

    def get_label_name(self, label_indx):
        """Takes class index and returns its text name for logging

        Args:
            label_indx (int): class index

        Returns:
            str: class name corresponding to input index
        """
        if not (self.is_domain_shift):
            return super().get_label_name(label_indx)
        if label_indx not in self.map_labels:
            raise ValueError("Label {} not in {} dataset".format(label_indx, self.name))
        return self.map_labels[label_indx]

    def prepare_data(self):
        """Prepare and download the dataset"""
        # inspired by https://github.com/cemsaz/city-scapes-script/blob/master/download_data.sh
        fine_annotations_name = os.path.expanduser(
            os.path.join(self.data_dir, "gtFine_trainvaltest.zip")
        )
        leftImg8_name = os.path.expanduser(
            os.path.join(self.data_dir, "leftImg8bit_trainvaltest.zip")
        )
        # download dataset
        if not (os.path.isfile(fine_annotations_name)) or not (
            os.path.isfile(leftImg8_name)
        ):
            session = requests.Session()
            login_data = {
                "username": self.username,
                "password": self.password,
                "submit": "Login",
            }
            session.post("https://www.cityscapes-dataset.com/login/", login_data)
        if not (os.path.isfile(fine_annotations_name)):
            fine_annotation_url = (
                "https://www.cityscapes-dataset.com/file-handling/?packageID=1"
            )
            fine_annotation = session.get(fine_annotation_url, allow_redirects=True)
            open(fine_annotations_name, "wb").write(fine_annotation.content)

        if not (os.path.isfile(leftImg8_name)):
            leftImg8_url = (
                "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
            )
            leftImg8 = session.get(leftImg8_url, allow_redirects=True)
            open(leftImg8_name, "wb").write(leftImg8.content)

    def _post_process_scenario(self, scenario):
        """Post processing step to add true targets in domain shift setup

        Args:
            scenario (Continuum.Scenario): output scenario after udpating targets
        """
        if not (self.is_domain_shift):
            return
        dataset = scenario.dataset
        original_targets = np.array(
            scenario.cl_dataset.get_base_dataset().targets
        ).squeeze()
        scenario.dataset = (dataset[0], original_targets, dataset[2])

    def get_current_task_classes(self):
        """Returns list of classes in the current task

        Returns:
            list: list of classes trained in the current task
        """
        if not (self.is_domain_shift):
            return super().get_current_task_classes()
        return range(self.num_classes)

    def setup(self, stage=None):
        """Assign train/val datasets for use in dataloaders

        Args:
            stage (str, optional): can be fit for training, test for test. Defaults to None.
        """
        if stage == "fit" or stage is None:
            # concatenate train and val dataset
            train_dataset = Cityscapes(
                self.data_dir,
                mode="fine",
                target_type="semantic",
                split="train",
            )
            val_dataset = Cityscapes(
                self.data_dir,
                mode="fine",
                target_type="semantic",
                split="val",
            )
            if self.continual and self.is_domain_shift:
                all_dataset = train_dataset
                all_dataset.targets += val_dataset.targets
                all_dataset.images += val_dataset.images
                # now split onto a validation and train
                self.train_dataset = BaseSegmentationDataset(
                    all_dataset,
                    transforms=self.train_transform,
                )
                self.val_dataset = None
            elif self.continual:
                self.train_dataset = BaseSegmentationDataset(
                    train_dataset,
                    transforms=self.train_transform,
                )
                self.val_dataset = BaseSegmentationDataset(
                    val_dataset,
                    transforms=self.test_transform,
                )
            else:
                train_dataset.transforms = self.train_transform
                val_dataset.transforms = self.test_transform
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset
                self.test_dataset = None
        if stage == "test" or stage is None:
            self.test_dataset = None
        self.update_continuum_datasets(
            stage,
            CustomDataset=CityscapeDomainDataset
            if self.is_domain_shift
            else PyTorchDataset,
        )

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
        if not (self.is_domain_shift):
            return super().get_dataloader(dataset, shuffle, with_prev_tasks, drop_last)
        num_workers = self.num_workers if num_workers is None else num_workers
        current_dataset = dataset
        if self.continual:
            if self._tmp_train is None:
                val_dataset = deepcopy(dataset)
                val_dataset.train = False
                val_dataset = self._get_continual_dataset(val_dataset, with_prev_tasks)
                current_dataset = self._get_continual_dataset(dataset, with_prev_tasks)
                train_length = int(0.8 * len(current_dataset))
                val_length = len(current_dataset) - train_length
                shuffle_split = ShuffleSplit(
                    n_splits=1,
                    test_size=val_length,
                    train_size=train_length,
                    random_state=self.seed,
                )
                train_indices, val_indices = next(
                    shuffle_split.split(range(train_length + val_length))
                )
                self._tmp_train = subset(current_dataset, train_indices)
                self._tmp_val = subset(val_dataset, val_indices)

            elif not (shuffle):
                current_dataset = self._tmp_val
                current_dataset.transforms = self.test_transform

            if shuffle:
                current_dataset = self._tmp_train
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
        self._tmp_train = None
        self._tmp_val = None
        return self.get_dataloader(self.train_dataset, drop_last=True)

    def val_dataloader(self):
        """Get validation data loader

        Returns:
            torch.dataloader: val dataloader used in the current step
        """
        return self.get_dataloader(self.val_dataset, shuffle=False)

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
            ClassIncrementalScenario: Class used to initialize the scenario
        Returns:
            Continuum: Current class incremental scenario
        """
        return super()._create_scenario(
            dataset,
            with_prev_tasks,
            ClassIncrementalScenario=CityScapeDomainScenario
            if self.is_domain_shift
            else CityScapeScenario,
        )
