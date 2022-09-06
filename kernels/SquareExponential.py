import math
from typing import List
import numpy as np


class SquaredExponentialKernel:
    
    def __init__(self, length = 1, sigma_f = 1):

        '''
            Take the minimum and maxiumum of the length and sigma values and run it through optimization and output the best results 

            Sigma   : Standard Deviation (vertical scale) -> 
            Length  : Length (horizontal scale)
            Sigma_y : Noise
        '''

        self.length = length
        self.sigma_f = sigma_f


    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:

        return float(self.sigma_f * np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) / (2 * self.length**2)))