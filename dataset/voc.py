# adapted from https://github.com/VainF/DeepLabV3Plus-Pytorch/ to support train augmentation not available in torchvision
import os
import torch.utils.data as data
from typing import List
from training.utils import download_extract
from PIL import Image
from torchvision.datasets.utils import download_url

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": "VOCdevkit/VOC2012",
    },
    "2011": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        "filename": "VOCtrainval_25-May-2011.tar",
        "md5": "6c3384ef61512963050cb5d687e5bf1e",
        "base_dir": "TrainVal/VOCdevkit/VOC2011",
    },
    "2010": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "filename": "VOCtrainval_03-May-2010.tar",
        "md5": "da459979d0c395079b5c75ee67908abb",
        "base_dir": "VOCdevkit/VOC2010",
    },
    "2009": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        "filename": "VOCtrainval_11-May-2009.tar",
        "md5": "59065e4b188729180974ef6572f6a212",
        "base_dir": "VOCdevkit/VOC2009",
    },
    "2008": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "2629fa636546599198acfcfbfcf1904a",
        "base_dir": "VOCdevkit/VOC2008",
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "base_dir": "VOCdevkit/VOC2007",
    },
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(
        self, root, year="2012", image_set="train", download=False, transform=None
    ):

        is_aug = False
        if year == "2012_aug":
            is_aug = True
            year = "2012"

        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]["url"]
        self.filename = DATASET_YEAR_DICT[year]["filename"]
        self.md5 = DATASET_YEAR_DICT[year]["md5"]
        self.transform = transform

        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]["base_dir"]
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, "JPEGImages")

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if is_aug and image_set == "train":
            mask_dir = os.path.join(voc_root, "SegmentationClassAug")
            if not (os.path.exists(mask_dir)):
                download_extract(
                    "http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip",
                    voc_root,
                    "SegmentationClassAug.zip",
                    "55b96877e788ccc7f733e6a2a541a4ad",
                )
                download_url(
                    "https://raw.githubusercontent.com/VainF/DeepLabV3Plus-Pytorch/master/datasets/data/train_aug.txt",
                    voc_root,
                    "train_aug.txt",
                )
            assert os.path.exists(
                mask_dir
            ), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join(voc_root, "train_aug.txt")
        else:
            mask_dir = os.path.join(voc_root, "SegmentationClass")
            splits_dir = os.path.join(voc_root, "ImageSets/Segmentation")
            split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"'
            )

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.targets = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @property
    def masks(self) -> List[str]:
        return self.targets
