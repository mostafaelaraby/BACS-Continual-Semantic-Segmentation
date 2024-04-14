from torch.utils.data import Dataset
from PIL import Image


class BaseSegmentationDataset(Dataset):
    """Wrapper on top of any base dataset to add specific segmentation transformation"""

    def __init__(self, base_dataset, transforms=None) -> None:
        self.base_dataset = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, mask = self.base_dataset[index]
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask


class BaseSubset(Dataset):
    """Wrapper on top of any base dataset to add specific segmentation transformation"""

    def __init__(
        self, base_dataset, transforms=None, target_trsf=None, use_transform=True
    ) -> None:
        self.base_dataset = base_dataset
        self.transforms = transforms
        self.use_target_trsf = target_trsf is not None
        self.use_transform = use_transform
        self.target_trsf = target_trsf

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        if not (self.use_transform):
            return self.base_dataset[index]
        img, mask = self.base_dataset[index]
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        if self.use_target_trsf:
            mask = self.target_trsf(mask)
        return img, mask


class BaseBufferDataset(Dataset):
    """Wrapper on top of any base dataset to add specific segmentation transformation"""

    def __init__(self, base_dataset, transforms=None, target_trsf=None) -> None:
        self.base_dataset = base_dataset
        self.transforms = transforms
        self.target_trsf = target_trsf
        self.use_transforms = transforms is not None
        self.use_target_trsform = target_trsf is not None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, mask = self.base_dataset[index]
        img = Image.open(img).convert("RGB")
        mask = Image.open(mask)
        if self.use_transforms:
            img, mask = self.transforms(img, mask)
        if self.use_target_trsform:
            # takes only single index
            mask = self.target_trsf[index](mask)
        return img, mask


class BaseMemMapDataset(Dataset):
    def __init__(
        self, imgs_map, logits_map, n_classes, length=None, transforms=None
    ) -> None:
        self.imgs_map = imgs_map
        self.logits_map = logits_map
        self.n_classes = n_classes
        self.length = length
        self.transforms = transforms
        self.use_transforms = transforms is not None

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.imgs_map)

    def __getitem__(self, index):
        img = self.imgs_map[index]
        logits = self.logits_map[index]
        if self.use_transforms:
            # no need to transform logits just input image
            # we use only torchvision simple transformation
            img = self.transforms(img)
        # buffer to replay img and logits only
        return img, logits, self.n_classes[index]
