from .interfaces import Loss
import numpy as np

class MSE(Loss):
    def forward(self, prediction, target):
        self.error = prediction - target
        return 0.5*np.mean(np.sum(np.square(self.error), axis = 0))
    
    def backward(self):
        return np.mean(self.error, axis = 1)
