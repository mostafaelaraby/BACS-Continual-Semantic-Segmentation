from dataset import VOCSegmentation, BaseSegmentationDataset
from continuum.transforms import segmentation as transforms
from .base_datamodule import BaseDataModule
from omegaconf import DictConfig
import os


class VocDataModule(BaseDataModule):
    """Pascal Voc 2012 augmented data module
    """

    def __init__(self, train_conf: DictConfig, dataset: DictConfig):
        super(VocDataModule, self).__init__(train_conf, dataset, name="VOC")
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
        self.map_labels = {
            0: "background",
            1: "aeroplane",
            2: "bicycle",
            3: "bird",
            4: "boat",
            5: "bottle",
            6: "bus",
            7: "car",
            8: "cat",
            9: "chair",
            10: "cow",
            11: "diningtable",
            12: "dog",
            13: "horse",
            14: "motorbike",
            15: "person",
            16: "pottedplant",
            17: "sheep",
            18: "sofa",
            19: "train",
            20: "tvmonitor",
        }

    def _get_normalization_mean_std(self):
        """Override to return normalization used in order to de-normalize

        Returns:
            tuple: tyuple of mean and std normalization if no normalization then return None
        """
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def prepare_data(self):
        """Prepare and download the dataset
        """
        # download VOC data
        # train set
        # check if the file exists or not
        voc_path = os.path.join(os.path.expanduser(self.data_dir), "VOCdevkit")
        if not (os.path.isdir(voc_path)):
            VOCSegmentation(
                self.data_dir, download=True, image_set="train", year="2012_aug"
            )
            # # validation set
            VOCSegmentation(
                self.data_dir, download=True, image_set="trainval", year="2012_aug"
            )
            # # test set
            VOCSegmentation(
                self.data_dir, download=True, image_set="val", year="2012_aug"
            )

    def setup(self, stage=None):
        """  Assign train/val datasets for use in dataloaders

        Args:
            stage (str, optional): can be fit for training, test for test. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = BaseSegmentationDataset(
                VOCSegmentation(
                    self.data_dir, download=False, image_set="train", year="2012_aug"
                ),
                transforms=self.train_transform,
            )
            self.val_dataset = BaseSegmentationDataset(
                VOCSegmentation(
                    self.data_dir, download=False, image_set="val", year="2012_aug"
                ),
                transforms=self.test_transform,
            )
        if stage == "test" or stage is None:
            # In Pascal VOC we rely only on val subset, trainval is used for warmup
            # but not used in continual semantic segmentation
            self.test_dataset = None
        self.update_continuum_datasets(stage)
