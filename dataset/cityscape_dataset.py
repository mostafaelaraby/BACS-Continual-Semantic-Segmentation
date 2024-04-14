from typing import List, Union
from continuum import SegmentationClassIncremental
import torchvision as tv
from PIL import Image
import numpy as np
from continuum.download import ProgressBar
import multiprocessing
import os
from training.utils import TransformLabel

id_to_trainid = {
    0: 0,
    1: 0,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 0,
    7: 1,  # road
    8: 2,  # sidewalk
    9: 0,
    10: 0,
    11: 3,  # building
    12: 4,  # wall
    13: 5,  # fence
    14: 0,
    15: 0,
    16: 0,
    17: 6,  # pole
    18: 0,
    19: 7,  # traffic light
    20: 8,  # traffic sign
    21: 9,  # vegetation
    22: 10,  # terrain
    23: 11,  # sky
    24: 12,  # person
    25: 13,  # rider
    26: 14,  # car
    27: 15,  # truck
    28: 16,  # bus
    29: 0,
    30: 0,
    31: 17,  # train
    32: 18,  # motorcycle
    33: 19,  # bicycle
    -1: 255,
}


map_labels = {
    0: "background",
    1: "road",
    2: "sidewalk",
    3: "building",
    4: "wall",
    5: "fence",
    6: "pole",
    7: "traffic light",
    8: "traffic sign",
    9: "vegetation",
    10: "terrain",
    11: "sky",
    12: "person",
    13: "rider",
    14: "car",
    15: "truck",
    16: "bus",
    17: "train",
    18: "motorcycle",
    19: "bicycle",
    -1: "license plate",
    255: "unknown",
}


class CityScapeScenario(SegmentationClassIncremental):
    def _get_label_transformation(self, task_index: Union[int, List[int]]):
        """Returns the transformation to apply on the GT segmentation maps.

        :param task_index: The selected task id.
        :return: A pytorch transformation.
        """
        if isinstance(task_index, int):
            task_index = [task_index]
        if not self.train:
            # In testing mode, all labels brought by previous tasks are revealed
            task_index = list(range(max(task_index) + 1))

        if self.mode in ("overlap", "disjoint"):
            # Previous and future (for disjoint) classes are hidden
            labels = self._get_task_labels(task_index)
        elif self.mode == "sequential":
            # Previous classes are not hidden, no future classes are present
            labels = self._get_task_labels(list(range(max(task_index) + 1)))
        else:
            raise ValueError(f"Unknown mode={self.mode}.")

        inverted_order = {label: self.class_order.index(label) + 1 for label in labels}
        inverted_order[255] = 255

        masking_value = 0
        if not self.train:
            if self.test_background:
                inverted_order[0] = 0
            else:
                masking_value = 255

        return  TransformLabel(id_to_trainid, 255, inverted_order,masking_value)

    def _setup(self, nb_tasks: int) -> int:
        """Setups the different tasks."""
        x, y, _ = self.cl_dataset.get_data()
        self.class_order = (
            self.class_order
            or self.cl_dataset.class_order
            or list(range(1, self._nb_classes + 1))
        )

        # For when the class ordering is changed,
        # so we can quickly find the original labels
        def class_mapping(c):
            if c in (0, 255):
                return c
            return self.class_order[c - 1]

        self._class_mapping = np.vectorize(class_mapping)

        self._increments = self._define_increments(
            self.increment, self.initial_increment, self.class_order
        )

        # Checkpointing the indexes if the option is enabled.
        # The filtering can take multiple minutes, thus saving/loading them can
        # be useful.
        if self.save_indexes is not None and os.path.exists(self.save_indexes):
            print(f"Loading previously saved indexes ({self.save_indexes}).")
            t = np.load(self.save_indexes)
        else:
            print("Computing indexes, it may be slow!")
            t = _filter_images(y, self._increments, self.class_order, self.mode)
            if self.save_indexes is not None:
                np.save(self.save_indexes, t)

        assert len(x) == len(y) == len(t) and len(t) > 0

        self.dataset = (x, y, t)

        return len(self._increments)


def _filter_images(
    paths: Union[np.ndarray, List[str]],
    increments: List[int],
    class_order: List[int],
    mode: str = "overlap",
) -> np.ndarray:
    """Select images corresponding to the labels.

    Strongly inspired from Cermelli's code:
    https://github.com/fcdl94/MiB/blob/master/dataset/utils.py#L19

    :param paths: An iterable of paths to gt maps.
    :param increments: All individual increments.
    :param class_order: The class ordering, which may not be [1, 2, ...]. The
                        background class (0) and unknown class (255) aren't
                        in this class order.
    :param mode: Mode of the segmentation (see scenario doc).
    :return: A binary matrix representing the task ids of shape (nb_samples, nb_tasks).
    """
    indexes_to_classes = []
    pb = ProgressBar()

    with multiprocessing.Pool(min(8, multiprocessing.cpu_count())) as pool:
        for i, classes in enumerate(pool.imap(_find_classes, paths), start=1):
            indexes_to_classes.append(classes)
            if i % 100 == 0:
                pb.update(None, 100, len(paths))
        pb.end(len(paths))

    t = np.zeros((len(paths), len(increments)))
    accumulated_inc = 0

    for task_id, inc in enumerate(increments):
        labels = class_order[accumulated_inc : accumulated_inc + inc]
        old_labels = class_order[:accumulated_inc]
        all_labels = labels + old_labels + [0, 255]

        for index, classes in enumerate(indexes_to_classes):
            if mode == "overlap":
                if any(c in labels for c in classes):
                    t[index, task_id] = 1
            elif mode in ("disjoint", "sequential"):
                if any(c in labels for c in classes) and all(
                    c in all_labels for c in classes
                ):
                    t[index, task_id] = 1
            else:
                raise ValueError(f"Unknown mode={mode}.")

        accumulated_inc += inc

    return t


def _find_classes(path: str) -> np.ndarray:
    """Open a ground-truth segmentation map image and returns all unique classes
    contained.

    :param path: Path to the image.
    :return: Unique classes.
    """
    classes = np.unique(np.array(Image.open(path)).reshape(-1))
    return np.unique(
        np.array([id_to_trainid.get(class_idx, 255) for class_idx in classes])
    )
