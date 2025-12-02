#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:08
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   dataloader4torch.py
# @Desc     :   

from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TorchDataLoader:
    """ A custom PyTorch DataLoader class for handling TorchDataset """

    def __init__(self, dataset: Dataset, batch_size: int = 32, is_shuffle: bool = True):
        """ Initialise the TorchDataLoader class
        :param dataset: the TorchDataset or Dataset to load data from
        :param batch_size: the number of samples per batch
        :param is_shuffle: whether to shuffle the data at every epoch
        """
        self._dataset: Dataset = dataset
        self._batches: int = batch_size
        self._is_shuffle: bool = is_shuffle

        self._loader: DataLoader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batches,
            shuffle=self._is_shuffle,
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """ Return a single (feature, label) pair or a batch via slice """
        if not isinstance(index, int):
            raise TypeError(f"Invalid index type: {type(index)}")
        return self._dataset[index]

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __repr__(self):
        return (f"TorchDataLoader(dataset={self._dataset}, "
                f"batch_size={self._batches}, "
                f"shuffle={self._is_shuffle})")


if __name__ == "__main__":
    pass
