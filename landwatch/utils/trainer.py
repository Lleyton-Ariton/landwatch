import os

import ray

import torch
import torch.nn as nn

from landwatch.model import UNet, DiceLoss
from landwatch.utils.preprocessing.dataset import SegmentationDataset

from torch.utils.data import DataLoader
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch import TorchTrainer

from typing import *


class SegmentationOperator(TrainingOperator):

    def setup(self, config: Dict):
        train_data = DataLoader(
            dataset=config['train_dataset'],
            shuffle=config.get('shuffle', True),
            batch_size=config['batch_size']
        )

        if config.get('test_dataset', True):
            test_data = None

        else:
            test_data = DataLoader(
                dataset=config['test_dataset'],
                shuffle=config.get('shuffle', True),
                batch_size=config['batch_size']
            )

        model = UNet(config['in_channels'], config['out_channels'])

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.get('lr', 1e4))
        criterion = config.get('criterion', DiceLoss(smooth=1.0))

        self.model, self.optimizer, self.criterion = self.register(
            models=model, optimizers=optimizer, criterion=criterion
        )

        self.register_data(train_loader=train_data, validation_loader=test_data)


class DefaultTorchTrainer:

    def __init__(self, config: Dict):
        self.config = config

        self.__use_gpu = self.config.get('use_gpu', True)
        if not torch.cuda.is_available():
            self.__use_gpu = False

        self.__train_data = DataLoader(
            dataset=config['train_dataset'],
            shuffle=config.get('shuffle', True),
            batch_size=config['batch_size']
        )

        if config.get('test_dataset', True):
            self.__test_data = None

        else:
            self.__test_data = DataLoader(
                dataset=config['test_dataset'],
                shuffle=config.get('shuffle', True),
                batch_size=config['batch_size']
            )

        self.__model = UNet(config['in_channels'], config['out_channels'])

        self.__model.train()
        if self.__use_gpu:
            self.__model.cuda()

        self.__optimizer = torch.optim.Adam(
            params=self.__model.parameters(),
            lr=config.get('lr', 1e4)
        )

        self.__criterion = config.get('criterion', DiceLoss(smooth=1.0))

    def train(self):
        for i, data in enumerate(self.__train_data):
            inputs, mask = data

            if self.__use_gpu:
                inputs.cuda(), mask.cuda()

                self.__criterion.cuda()

            self.__optimizer.zero_grad()

            outputs = self.__model

            loss = self.__criterion(outputs, mask)
            loss.backward()

            self.__optimizer.step()

    def get_model(self) -> nn.Module:
        self.__model.cpu()

        return self.__model

    def save(self, fp: Union[str, os.PathLike, BinaryIO]):
        self.__model.cpu()

        torch.save(self.__model, fp)
