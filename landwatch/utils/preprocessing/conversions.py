import torch
import torch.nn as nn

import numpy as np


def convert_img_to_input(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255


def convert_output_to_img(output: torch.Tensor) -> np.ndarray:
    return (output.detach().squeeze().permute(1, 2, 0) * 255).numpy().astype(np.uint8)
