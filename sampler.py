import numpy as np
import random
from typing import List, Tuple

import torch
from torch.utils.data import Sampler, Dataset


def LD2DT(LD):
    return {k: torch.stack([dic[k] for dic in LD]) for k in LD[0]}


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        # n_query_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        reptile_step: int = 3,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        # self.n_query_way = n_query_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.reptile_step = reptile_step
        self.replacement = False

        self.indices_per_label = {}

        if "label" in dataset.data.keys():
            for index, label in enumerate(dataset.data["label"].tolist()):
                if label in self.indices_per_label.keys():
                    self.indices_per_label[label].append(index)
                else:
                    self.indices_per_label[label] = [index]
        else:
            self.indices_per_label[0] = range(len(dataset))
            self.replacement = True

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.indices_per_label[label],
                            self.reptile_step * (self.n_shot + self.n_query),
                        )
                    )
                    for label in (
                        random.choices(
                            list(self.indices_per_label.keys()), k=self.n_way
                        )
                        if self.replacement
                        else random.sample(self.indices_per_label.keys(), self.n_way)
                    )
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            list({
                support: {key: Tensor for key in input_data},
                query: {key: Tensor for key in input_data}
            }) with length of reptile_step
        """
        if "label" in input_data[0].keys():
            input_data.sort(key=lambda item: item["label"])

        input_data = LD2DT(input_data)

        def split_tensor(tensor):
            """
            Function to split the input tensor into a list of support & query data with
            the length of reptile_step
            Args:
                tensor: input tensor (number of samples) x (data dimension)
            Returns:
                list([
                    Tensor((n_way * n_shot) x (data dimension)),
                    Tensor((n_way * n_query) x (data dimension))
                ]) with the length of reptile_step
            """
            tensor = tensor.reshape(
                (
                    self.n_way,
                    self.reptile_step * (self.n_shot + self.n_query),
                    *tensor.shape[1:],
                )
            )
            tensor_list = torch.chunk(tensor, self.reptile_step, dim=1)
            tensor_list = [
                [
                    split.flatten(end_dim=1)
                    for split in torch.split(item, [self.n_shot, self.n_query], dim=1,)
                ]
                for item in tensor_list
            ]

            return tensor_list

        data = {k: split_tensor(v) for k, v in input_data.items()}
        data = [
            {
                key: {k: v[i][j] for k, v in data.items()}
                for j, key in enumerate(["support", "query"])
            }
            for i in range(self.reptile_step)
        ]

        return data


class PrototypicalBatchSampler(object):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be inferred from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = (
            np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        )
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[
                label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]
            ] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        """
        yield a batch of indexes
        """
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = (
                    torch.arange(len(self.classes)).long()[self.classes == c].item()
                )
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations

