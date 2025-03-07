import numpy as np
import random

# Decorator to check for NaN values in numpy arrays sent to and returned from functions
# Used to help debug 
def nanlog(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        has_nan = None

        types = [type(arg) for arg in args]
        ndarrays = list(filter(lambda x: type(x) == "ndarray", types))

        for ndarray in ndarrays:
            has_nan = np.isnan(ndarray).any()
            if has_nan:
                break

        if has_nan:
            print(f"Array in func '{func}' contains NaN: {args}")

        has_nan = None

        if type(result) == "ndarray":
            has_nan = np.isnan(result).any()

        if has_nan:
            print(f"Array from func '{func}' contains NaN: {result}")

        return result

    return wrapper

class linear():
    def __init__(self, n: int, m: int):
        self.A = np.random.uniform(-1, 1, (n, m))
        self.b = np.random.rand(1, int(m))
        self.x = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        self.x = x 
        x = np.dot(x, self.A) + self.b
        return x

    # Calculate and apply gradient for this layer then pass on new derivative to next layer 
    def backward(self, d, lr, l):
        d = np.atleast_2d(d)
        grad_A = self.x.T @ d
        grad_b = d.sum(axis=0, keepdims=True)
        
        #print(f"d{l} : {d}")
        dx = d @ self.A.T

        self.A -= grad_A * lr
        self.b -= grad_b * lr

        return dx

class ReLU():
    def forward(self, z):
        self.mask = (z > 0).astype(float)
        return np.maximum(z, 0)

    def backward(self, d):
        return d * self.mask

    def __call__(self, z):
        return self.forward(z)

# Neural Network base class
# Defines the general structure of a neural network and all essential functionality
class net():
    def __init__(self, n: int, h: int, m: int):
        self.fc1 = linear(n, h)
        self.relu1 = ReLU()
        self.fc2 = linear(h, h)
        self.relu2 = ReLU()
        self.fc3 = linear(h, m)

    @nanlog
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        #x = self.relu1(self.fc1(x))
        #x = self.relu2(self.fc2(x))
        #x = self.softmax(self.fc3(x))
        return x

    # Uses stochastic gradient descent and applies loss
    @nanlog
    def gradient(self, d, lr):
        # Each call to Linear.backward() calculates the gradient of the layer, applies it, and then returns it so that the next layer can use it as the basis for its own gradient calculations
        d = self.fc3.backward(d, lr, 3)
        #d = self.relu2.backward(d)
        #d = self.fc2.backward(d, lr, 2)
        #d = self.relu1.backward(d)
        d = self.fc1.backward(d, lr, 1)

    @staticmethod
    @nanlog
    def softmax(z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        sigma = e / e.sum(axis=1, keepdims=True)

        return sigma 

    @staticmethod
    @nanlog
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def least_squares(x, y):
        s = y - x
        return np.square(s).sum()

    @staticmethod
    @nanlog
    def cross_entropy(x, y):
       return -np.sum((1 - x) * np.log(y + 1e-8)) 
