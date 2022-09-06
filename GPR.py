import numpy as np
from kernels.SquareExponential import SquaredExponentialKernel
from plots import *
from utils.optimize import optimize
from base_gp import Base_GP


class GPR(Base_GP):


    '''

        Child Class of Base_GP
    
    '''


    def __init__(self, data_x, data_y):
        super().__init__(data_x, data_y)

        # main test data

        self.data_x = data_x
        self.data_y = data_y

        # optimizes automatically

        self.hyperparameters = optimize(self.data_x, self.data_y) # function
        self._model = Base_GP(
                            data_x              = self.data_x,
                            data_y              = self.data_y,
                            covariance_function = SquaredExponentialKernel(length=self.hyperparameters['length_optimal'], sigma_f=1),
                            white_noise_sigma   = self.hyperparameters['sigma_optimal']
                        )

        
    def re_train(self):

        '''
            Purpose: Depending on the siutation. You might want to reoptimize your kernel parameters to keep it fresh. 
        '''


        self.hyperparameters = optimize(self.data_x, self.data_y)


    def predict_GPR(self, at_values : np.array):

        '''
            Role: It uses the base_GP class predict to give us the mean and stores results (mean, variance, and covariance)
        '''


        return self._model.predict(at_values=at_values)





def test_function(x: np.array) -> np.array:
    return np.sin(x)


if __name__ == '__main__':

    # objective function

    x_values = np.arange(0, 5, 0.3)
    y_values = test_function(x_values)
    gpr = GPR(x_values,y_values)


    # test samples 

    x = np.arange(-1, 10, 0.05)
    mean = gpr.predict_GPR(x)
    std = np.sqrt(gpr._model._memory['variance'])

    data = []

    for i in range(1, 4):
        data.append(
            uncertainty_area_scatter(
                x_lines=x,
                y_lower=mean - i * std,
                y_upper=mean + i * std,
                name=f"mean plus/minus {i}*standard deviation"))

    data += [
        line_scatter(x_lines=x, y_lines=mean),
        dot_scatter(x_dots=x_values, y_dots=y_values)
    ]

    fig = go.Figure(data=data)
    fig = update_layout_of_graph(fig,title='GPR with parameters trained')

    fig.show()

    

