from .interfaces import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, n, m):
        self.W = np.asmatrix(np.random.rand(m,n))
    
    def forward(self, x, train = True):
        if train: self.x = np.asmatrix(x)
        return self.W @ x
    
    def backward(self, grad, lr):
        self.W -= lr * grad @ np.mean(self.x.T, axis = 0)
        return self.W.T @ grad

class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, x, train = True):
        if train: self.x = x
        return np.maximum(x, 0)

    def backward(self, grad, lr):
        return np.multiply(grad, np.mean(self.x > 0, axis = 1))

class Tanh(Layer):
    def __init__(self):
        pass

    def forward(self, x, train = True):
        if train: self.x = x
        return np.tanh(x)

    def backward(self, grad, lr):
        return np.multiply(grad, np.mean(1-np.square(np.tanh(self.x)), axis = 1))

class Sequential(Layer):
    def __init__(self, *args):
        self.layers = args
    
    def forward(self, x, train = True):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x
    
    def backward(self, grad, lr):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, lr)
        return grad

class LinearReLU(Layer):
    def __init__(self, n, m):
        self.layer = Sequential(Linear(n, m), ReLU())

    def forward(self, x, train = True):
        return self.layer.forward(x, train)

    def backward(self, grad, lr):
        return self.layer.backward(grad, lr)

class LinearTanh(Layer):
    def __init__(self, n, m):
        self.layer = Sequential(Linear(n, m), Tanh())

    def forward(self, x, train = True):
        return self.layer.forward(x, train)

    def backward(self, grad, lr):
        return self.layer.backward(grad, lr)
