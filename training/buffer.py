import torch
from hydra.utils import get_original_cwd
import os
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
from typing import Tuple
from shutil import copyfile
from scipy.special import softmax


class DatasetMap(Dataset):
    def __init__(
        self, size: int, data_size: tuple, data_type: str, path: str, name: str
    ) -> None:
        self.name = name
        self.size = size
        self.data_size = data_size
        self.path = path
        self.tmp_path = os.path.join(path, "tmp_{}".format(self.name))
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.tmp_path, exist_ok=True)
        self.data_type = data_type
        self.file_path, self.increment = self._get_full_path(name)
        self.data_map = np.memmap(
            self.file_path,
            dtype=data_type,
            mode="w+",
            shape=(self.size, *self.data_size),
        )
        # Initialize number of saved records to zero
        self.length = 0

    def _get_full_path(self, name, increment=0):
        # used to spawn different files for multi-processes
        path = os.path.join(self.path, "{}_{}.dat".format(name, increment))
        if os.path.exists(path):
            path, increment = self._get_full_path(name, increment=increment + 1)
        return path, increment

    def __getitem__(self, index) -> torch.Tensor:
        return self.data_map[index, ...]

    def __len__(self) -> int:
        return self.length

    def add(self, item, index):
        self.data_map[index] = item
        self.length += 1

    def extend(self, items, indices):
        for index, item in zip(indices, items):
            if index > 0:
                self.add(item, index)

    def _close(self, num_memmap):
        num_memmap.flush()
        if os.name == "nt":
            mm_file = num_memmap._mmap
            if mm_file is not None:
                mm_file.close()

    def change_data_size(self, new_data_size):
        self._close(self.data_map)
        original_path = self.file_path
        tmp_path, self.increment = self._get_full_path(
            self.name, increment=self.increment
        )
        copyfile(
            self.file_path,
            tmp_path,
        )
        self.data_map = np.memmap(
            original_path,
            dtype=self.data_type,
            mode="w+",
            shape=(self.size, *new_data_size),
        )
        tmp_original = np.memmap(
            tmp_path,
            dtype=self.data_type,
            mode="r",
            shape=(self.size, *self.data_size),
        )
        self.data_map[:, : self.data_size[0], ...] = tmp_original.copy()
        self.data_size = new_data_size
        if os.name == "nt":
            print("windows detected")
            self._close(tmp_original)
        try:
            os.remove(tmp_path)
        except:
            pass


class Buffer:
    def __init__(
        self,
        buffer_size,
        buffer_name,
        same_task=False,
        task_num=-1,
        transformations=None,
    ) -> None:
        self.buffer_name = buffer_name
        self.buffer_size = buffer_size
        self._logits_n_classes = np.zeros(buffer_size, dtype="uint8")
        self._task_id_list = np.zeros(buffer_size, dtype="uint8")
        self.dataset_map = None
        # to sample a batch from a specific previous task
        self.same_task = same_task
        self.task_num = task_num
        # seen examples so far
        self._num_seen_examples = 0
        self.transformations = transformations
        # Scores for balanced sampling
        self.importance_score = np.ones(buffer_size, dtype="float") * -1 * np.Inf
        self.balance_score = np.ones(buffer_size, dtype="float") * -1 * np.Inf
        self.scores = np.ones(buffer_size, dtype="float") * -1 * np.Inf
        self._existing_indices = np.full(buffer_size, False)
        # dictionary of number of occurence of a specific class
        self.labels = {}
        # store labels existing in each sample
        self._examples_labels = {}
        self.img_paths = {}
        self.target_paths = {}
        self.target_trsf = {}
        self.co_occurance_map = None

    def get_importance(self):
        if not ((self.importance_score != (-1 * np.Inf)).any()):
            return 10
        scores = np.median(
            -1 * self.importance_score[self.importance_score != (-1 * np.Inf)]
        )
        return scores

    def merge_scores(self, co_occurance_map=None):
        self.co_occurance_map = co_occurance_map
        # update balance score
        for current_data_index in self._examples_labels:
            self.balance_score[current_data_index] = min(
                [
                    self.labels[label]
                    for label in self._examples_labels[current_data_index]
                    if label != 0
                ]
            )
        # COndition on the absense of loss function / importance score
        scaling_factor = np.mean(abs(self.importance_score)) * np.mean(
            abs(self.balance_score)
        )
        norm_importance = self.importance_score / scaling_factor
        # 0.2 and 0.8 are better
        presoftscores = 0.3 * norm_importance + 0.7 * self.balance_score

        if presoftscores.max() - presoftscores.min() != 0:
            presoftscores = (presoftscores - np.min(presoftscores)) / (
                np.max(presoftscores) - np.min(presoftscores)
            )
        self.scores = presoftscores / np.sum(presoftscores)

    def functionalReservoir(self, N, m):
        if N < m:
            return N
        rn = np.random.randint(0, N)
        if rn < m:
            self.merge_scores()
            index = np.random.choice(range(m), p=self.scores, size=1)[0]
            return index
        else:
            return -1

    def update_task(self, task_num, new_class_size):
        self.task_num = task_num
        logits_saved = self.dataset_map is not None and "logits" in self.dataset_map
        if (
            logits_saved
            and new_class_size > self._logits_n_classes.max()
            and self.num_seen_examples > 0
        ):
            # update size.
            current_logits_shape = self.dataset_map["logits"].data_size
            self.dataset_map["logits"].change_data_size(
                [new_class_size, current_logits_shape[1], current_logits_shape[2]]
            )
        print("================== Existing Task list in buffer ===============")
        print(np.unique(self._task_id_list))

    @property
    def num_seen_examples(self):
        return self._num_seen_examples

    def _init_map(self, dict_data):
        self.dataset_map = {}
        for attr_str in dict_data:
            self.dataset_map[attr_str] = DatasetMap(
                self.buffer_size,
                dict_data[attr_str].shape[1:],
                str(dict_data[attr_str].dtype).split(".")[-1],
                os.path.join(get_original_cwd(), "mem_maps", self.buffer_name),
                attr_str,
            )

    def add_data(self, dict_data):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        dict_data: dictionary including the following defaults key, value
        :param examples: tensor containing the images
        :param logits: tensor containing the outputs of the network
        :return:
        """
        has_paths = "img_paths" in dict_data
        if has_paths:
            img_paths = dict_data["img_paths"]
            target_paths = dict_data["target_paths"]
            target_trsf = dict_data["target_trsf"]
            del dict_data["target_trsf"]
            del dict_data["img_paths"]
            del dict_data["target_paths"]

        has_loss_scores = "loss" in dict_data
        if self.dataset_map is None:
            self._init_map(dict_data)
        if "logits" in dict_data:
            current_n_classes = dict_data["logits"].shape[1]
        if "loss" in dict_data:
            loss_scores = dict_data["loss"].cpu().numpy()
            del dict_data["loss"]
        assert "examples" in dict_data
        n_examples = dict_data["examples"].shape[0]
        indices = []
        for index in range(n_examples):
            # update balance score
            current_data_index = self.functionalReservoir(
                self.num_seen_examples, self.buffer_size
            )
            if current_data_index >= 0:
                current_labels = dict_data["labels"].cpu().unique().numpy()
                # pop labels from selected index
                if self.num_seen_examples >= self.buffer_size:
                    for former_labels in self._examples_labels[current_data_index]:
                        if former_labels != 0 and former_labels in self.labels:
                            self.labels[former_labels] -= 1
                self._examples_labels[current_data_index] = current_labels
                # updating labels
                for label in current_labels:
                    if label == 0:
                        continue
                    if label in self.labels:
                        self.labels[label] += 1
                    else:
                        self.labels[label] = 1
                if "logits" in dict_data:
                    self._logits_n_classes[current_data_index] = current_n_classes
                if has_paths:
                    self.img_paths[current_data_index] = img_paths[index]
                    self.target_paths[current_data_index] = target_paths[index]
                    self.target_trsf[current_data_index] = target_trsf[index]
                self._task_id_list[current_data_index] = self.task_num
                self._existing_indices[current_data_index] = True
                indices.append(current_data_index)
                self.importance_score[current_data_index] = (
                    loss_scores[index] if has_loss_scores else -1 * np.Inf
                )
            self._num_seen_examples += 1

        for param_name, param_value in dict_data.items():
            assert param_name in self.dataset_map
            self.dataset_map[param_name].extend(param_value.cpu().numpy(), indices)

    def get_available_tasks(self):
        return np.unique(self._task_id_list)

    def _co_occurance_image_blend(self, labels, alpha=1.0, threshold=10):
        def _filter_array(input_array):
            return input_array[(input_array != 0) & (input_array != 255)]

        # we assume co_occurance_map is not None once this function is called
        batch_size = labels.shape[0]
        indices = np.zeros(batch_size, dtype=int)
        lamdas = np.zeros((batch_size, 1, 1, 1))
        # threshold can be {50, 40, 30, 20, 10}
        for first_indx, first_label in enumerate(labels):
            similarity_score = np.zeros(batch_size, dtype=int)
            mixed_category_list = np.zeros(batch_size, dtype=int)
            for second_idx, second_label in enumerate(labels):
                if first_indx == second_idx:
                    continue
                first_unique = _filter_array(np.unique(first_label))
                second_unique = _filter_array(np.unique(second_label))
                if len(first_unique) == 0 or len(second_unique) == 0:
                    continue
                similarity_score[second_idx] = np.sum(
                    self.co_occurance_map[first_unique, :][:, second_unique]
                )
                mixed_category_list[second_idx] = len(first_unique) + len(second_unique)
            indices[first_indx] = np.argmax(similarity_score)
            if mixed_category_list[indices[first_indx]] > threshold:
                lamdas[first_indx] = 0.9
            else:
                lamdas[first_indx] = np.random.beta(alpha, alpha)
        return lamdas, indices

    def _sample_indices(self, sample_size, same_task=False, task_num=None):
        if sample_size > self.num_seen_examples:
            sample_size = self.num_seen_examples
        sample_task_id = -1
        existing_indices = np.where(self._existing_indices)[0]
        if same_task:
            # do same task importance sampling based on number of classes in each
            sample_task_id = (
                np.random.choice(np.unique(self._task_id_list), size=1)[0]
                if task_num is None
                else task_num
            )
            all_indices = np.where(
                (self._task_id_list == sample_task_id) & self._existing_indices
            )[0]
            if all_indices.size >= sample_size:
                choice = np.random.choice(all_indices, size=sample_size, replace=False)
            else:
                choice = np.random.choice(
                    (
                        existing_indices
                        if self.num_seen_examples < self.buffer_size
                        else self.buffer_size
                    ),
                    size=(sample_size - all_indices.size),
                    replace=False,
                )
                choice = np.concatenate([choice, all_indices], axis=0)
        else:
            choice = np.random.choice(
                (
                    existing_indices
                    if self.num_seen_examples < self.buffer_size
                    else self.buffer_size
                ),
                size=sample_size,
                replace=False,
                # p= softmax(-1 * self.importance_score),
            )
        return choice, sample_task_id

    def get_data(
        self,
        size: int,
        return_indexes=False,
        same_task=False,
        task_num=None,
        mixup=False,
        device=None,
    ) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return: a dictionary of attributes
        """
        choice, task_id = self._sample_indices(
            size, same_task=same_task, task_num=task_num
        )
        ret_dict = {}
        do_mixup = self.co_occurance_map is not None and mixup
        if do_mixup:
            labels = self.dataset_map["labels"][choice]
            lamdas, indices = self._co_occurance_image_blend(labels)
            ret_dict["lamdas"] = lamdas
            ret_dict["indices"] = indices
        for dataset_name, dataset_map in self.dataset_map.items():
            if do_mixup and dataset_name == "labels":
                current_samples = labels
            else:
                current_samples = dataset_map[choice]
            if do_mixup and dataset_name == "examples":
                ret_dict[dataset_name] = torch.tensor(
                    lamdas * current_samples + lamdas * current_samples[indices]
                )
            else:
                ret_dict[dataset_name] = torch.tensor(current_samples, device=device)
        ret_dict["n_classes"] = self._logits_n_classes[choice]
        if self.transformations is not None:
            ret_dict["examples"] = self.transformations(ret_dict["examples"])
        ret_dict["task_id"] = task_id
        if not return_indexes:
            return ret_dict
        else:
            return ret_dict, choice

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False
