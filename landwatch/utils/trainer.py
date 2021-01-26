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
