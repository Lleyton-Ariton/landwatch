import ray
import torch

from landwatch.model import UNet, DiceLoss
from landwatch.utils.data.dataset import SegmentationDataset

from torch.utils.data import DataLoader
from ray.util.sgd.torch import TrainingOperator

from typing import *


class RayHandler:

    def __enter__(self, **kwargs):
        if ray.is_initialized():
            raise RuntimeError('Ray is already initialized')

        ray.init(**kwargs)

    def __exit__(self):
        if ray.is_initialized():
            ray.shutdown()


class SegmentationOperator(TrainingOperator):

    def setup(self, config: Dict):
        train_data = SegmentationDataset(root_dir=config['train_dir'])
        test_data = SegmentationDataset(root_dir=config['test_dir'])

        train_data = DataLoader(
            train_data,
            batch_size=config['batch_size'],
            shuffle=config.get('shuffle', True)
        )

        test_data = DataLoader(
            test_data,
            batch_size=config['batch_size'],
            shuffle=config.get('shuffle', True)
        )

        model = UNet(config['in_channels'], config['out_channels'])

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.get('lr', 1e4))
        criterion = DiceLoss(smooth=1.0)

        self.model, self.optimizer, self.criterion = self.register(
            models=model, optimizers=optimizer, criterion=criterion
        )

        self.register_data(train_loader=train_data, validation_loader=test_data)
