from .continuum_dataset import PyTorchDataset
from continuum import SegmentationClassIncremental
import torchvision as tv
import os
import numpy as np
from typing import List, Tuple, Union
from PIL import Image
from typing import Tuple
from training.utils import TransformLabel

# Converting the id to the train_id. Many objects have a train id at
# 255 (unknown / ignored).
# See there for more information:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
id_to_trainid = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,  # road
    8: 1,  # sidewalk
    9: 255,
    10: 255,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,  # pole
    18: 255,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255,
    30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 0,
}

map_labels_domain = {
    255: "unlabeled",
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    -1: "license plate",
}


city_to_id = {
    "aachen": 0,
    "bremen": 1,
    "darmstadt": 2,
    "erfurt": 3,
    "hanover": 4,
    "krefeld": 5,
    "strasbourg": 6,
    "tubingen": 7,
    "weimar": 8,
    "bochum": 9,
    "cologne": 10,
    "dusseldorf": 11,
    "hamburg": 12,
    "jena": 13,
    "monchengladbach": 14,
    "stuttgart": 15,
    "ulm": 16,
    "zurich": 17,
    "frankfurt": 18,
    "lindau": 19,
    "munster": 20,
}


class CityScapeDomainScenario(SegmentationClassIncremental):
    def _get_label_transformation(self, task_index: Union[int, List[int]]):
        """Returns the transformation to apply on the GT segmentation maps.

        :param task_index: The selected task id.
        :return: A pytorch transformation.
        """
        # shuffling of classes not supported in domain shift
        return TransformLabel(id_to_trainid, 255)


class CityscapeDomainDataset(PyTorchDataset):
    def __init__(
        self,
        data_path: str = "",
        dataset_type=None,
        dict_args: dict = ...,
        transformation=None,
        train=True,
    ):
        super().__init__(data_path, dataset_type, dict_args, transformation, train)
        self.domains = []
        # loop over the full base dataset to get domains
        self.update_domains()

    def update_domains(self):
        dataset = self.get_base_dataset()
        all_images = dataset.images
        all_targets = dataset.targets
        dataset.images = []
        dataset.targets = []
        tmp_dir = "tmp/"
        if not (os.path.isdir(tmp_dir)):
            os.makedirs(tmp_dir)
        for file_name, target_name in zip(all_images, all_targets):
            root_dir_split = file_name.split(os.sep)
            root_dir = (os.sep).join(root_dir_split[:-1])
            if isinstance(target_name, list):
                target_name = target_name[0]
            target_dir = (os.sep).join(target_name.split(os.sep)[:-1])
            file_name = root_dir_split[-1]
            city = root_dir_split[-2]
            city_path = os.path.join(tmp_dir, city + ".png")
            target_types = []
            for t in dataset.target_type:
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0],
                    dataset._get_target_suffix(dataset.mode, t),
                )
                target_types.append(os.path.join(target_dir, target_name))
            # save the domain image
            if not (os.path.isfile(city_path)):
                # read image
                example_image = np.array(Image.open(target_types[-1]))
                city_domain = np.zeros_like(example_image) + city_to_id[city]
                city_domain = Image.fromarray(city_domain)
                city_domain.save(city_path)
            dataset.images.append(os.path.join(root_dir, file_name))
            if len(target_types) == 1:
                dataset.targets.append(target_types[0])
            else:
                dataset.targets.append(target_types[0])
            self.domains.append(city_path)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = self.get_base_dataset()
        x, y = np.array(dataset.images), np.array(self.domains)
        return x, y, None
