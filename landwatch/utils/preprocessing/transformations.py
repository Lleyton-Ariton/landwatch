import cv2
import numpy as np

from typing import *


class FlipImageVR:
    
    def __call__(self,
                 inputs: Union[List[np.ndarray],
                               Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
                     
        return [cv2.flip(inputs[0], 0), cv2.flip(inputs[1], 0)]


class FlipImageHZ:
    
    def __call__(self,
                 inputs: Union[List[np.ndarray],
                               Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
                     
        return [cv2.flip(inputs[0], 1), cv2.flip(inputs[1], 1)]


class DoubleFlip:
    
    def __call__(self,
                 inputs: Union[List[np.ndarray],
                               Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
                     
        return [cv2.flip(inputs[0], -1), cv2.flip(inputs[1], -1)]
