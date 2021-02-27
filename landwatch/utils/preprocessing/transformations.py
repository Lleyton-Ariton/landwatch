import abc

import cv2
import numpy as np

from typing import *


class Transformation(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, inputs: Union[List[np.ndarray],
                               Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        raise NotImplementedError()


class FlipImageVR(Transformation):
    
    def __call__(self,
                 inputs: Union[List[np.ndarray],
                               Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
                     
        return [cv2.flip(inputs[0], 0), cv2.flip(inputs[1], 0)]


class FlipImageHZ(Transformation):
    
    def __call__(self,
                 inputs: Union[List[np.ndarray],
                               Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
                     
        return [cv2.flip(inputs[0], 1), cv2.flip(inputs[1], 1)]


class DoubleFlip(Transformation):
    
    def __call__(self,
                 inputs: Union[List[np.ndarray],
                               Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
                     
        return [cv2.flip(inputs[0], -1), cv2.flip(inputs[1], -1)]
