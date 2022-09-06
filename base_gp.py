import numpy as np
from kernels.SquareExponential import SquaredExponentialKernel
from plots import *
from utils.optimize import optimize


def cov_matrix(x1, x2, cov_function) -> np.array:
    return np.array([[cov_function(a, b) for a in x1] for b in x2])



'''

    BASELINE CLASS FOR GAUSSIAN PROCESSES 

    parent : Base_GP

    children:
    - GPR (GAUSSIAN PROCESS REGRESSION)
    - etc..

'''


class Base_GP:

    def __init__(self,
                 data_x: np.array,
                 data_y: np.array,
                 covariance_function = SquaredExponentialKernel(),
                 white_noise_sigma: float = 0):
        
        self.noise = white_noise_sigma
        self.data_x = data_x
        self.data_y = data_y
        self.covariance_function = covariance_function

        # Store the inverse of covariance matrix of input (+ machine epsilon on diagonal) since it is needed for every prediction
        self._inverse_of_covariance_matrix_of_input = np.linalg.inv(cov_matrix(data_x, data_x, self.covariance_function) + (3e-7 + self.noise) * np.identity(len(self.data_x)))

        # stores prior points
        self._memory = None

    def predict(self, at_values: np.array) -> np.array:

        '''
            Purpose: predicts output at new input values. Store the mean and covariance matrix in memory.
        '''

        k_lower_left = cov_matrix(self.data_x, at_values, self.covariance_function)                                                 # Covariance of prior points in conjuction with our input (help with matrix multiplication) 
        k_lower_right = cov_matrix(at_values, at_values, self.covariance_function)                                                  # Covariance of input data
        mean_at_values = np.dot(k_lower_left, np.dot(self.data_y, self._inverse_of_covariance_matrix_of_input.T).T).flatten()       # Mean
        cov_at_values = k_lower_right - np.dot(k_lower_left, np.dot(self._inverse_of_covariance_matrix_of_input, k_lower_left.T))   # Covariance

        # Adding value larger than machine epsilon to ensure positive semi definite
        cov_at_values = cov_at_values + 3e-7 * np.ones(np.shape(cov_at_values)[0])

        var_at_values = np.diag(cov_at_values)

        self._memory = {
            'mean': mean_at_values,
            'covariance_matrix': cov_at_values,
            'variance': var_at_values
        }

        return mean_at_values
