import torch
from torch.utils.data import Dataset

from torchvision.transforms import *


class SegmentationDataset(Dataset):

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

        pass  # Not Finished
