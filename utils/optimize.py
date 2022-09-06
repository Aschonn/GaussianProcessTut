import numpy as np
from scipy.optimize import minimize
from kernels.SquareExponential import SquaredExponentialKernel
from plots import *



def optimize(data_x, data_y): 

    def adjusted_log_likelihood_RBF_kernel(parameters) -> float:

        lengthscale = parameters[0]
        white_noise = parameters[1]
        kernel = SquaredExponentialKernel(length=lengthscale, sigma_f=white_noise)
        covariance_matrix = np.matrix([[kernel(a, b) for a in data_x] for b in data_x]) + (3e-7 + white_noise) * np.identity(len(data_x))
        inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        return np.asarray(
            np.dot(np.dot(data_y.T, inverse_covariance_matrix), data_y) +
            np.log(np.linalg.det(covariance_matrix))).flatten()[0]

    # For simplicity we did not included the jacobian and hessian matrix
    optimal_values = minimize(adjusted_log_likelihood_RBF_kernel, [1, 0],
                                                                        options={
                                                                            'disp': True,
                                                                            'maxiter': 100
                                                                        },
                                                                        bounds=((0.0001, None), (0.0001, None)))


    return {
        'length_optimal' : optimal_values.x[0],
        'sigma_optimal'  : optimal_values.x[1]
    }
