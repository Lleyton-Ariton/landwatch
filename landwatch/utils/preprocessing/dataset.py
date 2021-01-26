import os
import cv2

import ray

import torch
from torch.utils.data import Dataset

import numpy as np
import multiprocessing as mp
from landwatch.utils.multicore import RayHandler

from typing import *


class SegmentationDataset(Dataset):

    IGNORE_FILES = {
        '.DS_Store'
    }

    @classmethod
    def add_ignore_file(cls, file_name: str):
        cls.IGNORE_FILES.add(file_name)

    @classmethod
    def remove_ignore_file(cls, file_name: str):
        cls.IGNORE_FILES.remove(file_name)

    @staticmethod
    def __filter_ignore_files(file_names: List[str]) -> List[str]:
        return list(filter(lambda filename: filename not in SegmentationDataset.IGNORE_FILES, file_names))

    @staticmethod
    def __process_image(file_path: str, resize: Tuple[int, int]=(224, 224)) -> List[List[List]]:
        if '.DS_Store' in file_path:
            return []
        else:
            if 'sat' in file_path:
                image = cv2.resize(cv2.imread(file_path), resize)

                mask = cv2.resize(cv2.imread(
                    file_path.replace('jpg', 'png').replace('sat', 'mask')
                ), resize)

                return [image, mask]

            image = cv2.resize(cv2.imread(
                file_path.replace('png', 'jpg').replace('mask', 'sat')
            ), resize)

            mask = cv2.resize(cv2.imread(file_path), resize)

            return [image, mask]

    def __load_data(self, file_path: str, resize: Tuple[int, int] = (224, 224),
                    num_cpus: int = mp.cpu_count(), n_images: int = None) -> List:

        it = ray.util.iter.from_items(
            self.__filter_ignore_files(os.listdir(file_path)),
            num_shards=num_cpus
        ).for_each(lambda filename: self.__process_image(f'{file_path}/{filename}', resize=resize))

        it = it.gather_async()

        return torch.tensor(it.take(len(os.listdir(file_path)) if n_images is None else n_images)).split(1, dim=1)

    def __init__(self, root_dir: str, resize: Tuple[int, int]=(224, 224),
                 transformations: List[Callable]=None, num_cpus: int=mp.cpu_count(), n_images: int=None):

        super().__init__()

        self.root_dir = root_dir
        self.resize = resize

        self.transformations = transformations
        if self.transformations is None:
            self.transformations = []

        self.num_cpus = num_cpus
        self.n_images = n_images

        self.x_data, self.y_data = self.__load_data(self.root_dir,
                                                    resize=self.resize,
                                                    num_cpus=self.num_cpus,
                                                    n_images=self.n_images)

        for transformation in self.transformations:
            self.x_data = transformation(self.x_data)

        self.x_data, self.y_data = self.x_data.squeeze(), self.y_data.squeeze()
        self.x_data, self.y_data = self.x_data.permute(0, 3, 1, 2), self.y_data.permute(0, 3, 1, 2)
        self.x_data, self.y_data = self.x_data.float(), self.y_data.float()

        self.x_data /= 255
        self.y_data /= 255

        if self.x_data.shape[0] != self.y_data.shape[0]:
            raise RuntimeError(f'Size mismatch! \'x_data has\' length {self.x_data.shape[0]} '
                               f'while \'y_data\' has length {self.y_data.shape[0]}')

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[item], self.y_data[item]


def create_dataset(file_path: str, create_handler: bool=False, **dataset_params) -> SegmentationDataset:
    if create_handler:
        with RayHandler():
            dataset = SegmentationDataset(file_path, **dataset_params)

        return dataset

    return SegmentationDataset(file_path, **dataset_params)
