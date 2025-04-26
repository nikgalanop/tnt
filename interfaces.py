import abc

class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x, train = True):
        pass

    @abc.abstractmethod
    def backward(self, grad):
        pass

    def predict(self, x):
        return self.forward(x, train = False)
    
class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, prediction, target):
        pass

    @abc.abstractmethod
    def backward(self):
        pass
