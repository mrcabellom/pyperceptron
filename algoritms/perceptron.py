import numpy as np

class Perceptron(object):
    """Binary Perceptron"""

    def __init__(self, eta=0.1, bias=0):
        self.eta = eta
        self.bias = bias
        self.weights = None
        self.errors = 0

    def fit(self, x, y):
        """Fit perceptron"""
        self.weights = np.zeros(x.shape[1])
        for index_x, target in zip(x, y):
            update = self.eta * (target - self.predict(index_x))
            self.weights += update * index_x
            self.errors += int(update != 0.0)
        return self

    def __net_input(self, x):
        return np.dot(x, self.weights) + self.bias

    def predict(self, x):
        return np.where(self.__net_input(x) > 0 , 1, 0)
