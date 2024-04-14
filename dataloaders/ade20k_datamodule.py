from .base_datamodule import BaseDataModule
from dataset import BaseSegmentationDataset
from continuum.transforms import segmentation as transforms
from omegaconf import DictConfig
from dataset import ADE20K
import os


class ADE20kDataModule(BaseDataModule):
    """Cityscape datamodule for pt lightning
    """    
    def __init__(self, train_conf: DictConfig, dataset: DictConfig):
        super(ADE20kDataModule, self).__init__(train_conf, dataset, name="CityScape")
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
        # map index to label
        self.map_labels = ADE20K.MAP_LABELS

    def prepare_data(self):
        """Prepare and download the dataset
        """
        ade_path = os.path.join(os.path.expanduser(self.data_dir), "ADEChallengeData2016")
        if not (os.path.isdir(ade_path)):
            ADE20K(
                self.data_dir, download=True, split="train"
            )

    def setup(self, stage=None):
        """  Assign train/val datasets for use in dataloaders

        Args:
            stage (str, optional): can be fit for training, test for test. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = BaseSegmentationDataset(
                ADE20K(
                    self.data_dir, split="train",
                ),
                transforms=self.train_transform,
            )
            self.val_dataset = BaseSegmentationDataset(
                ADE20K(
                    self.data_dir, split="val",
                ),
                transforms=self.test_transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = None
        self.update_continuum_datasets(stage)
