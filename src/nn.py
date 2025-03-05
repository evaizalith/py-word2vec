import numpy as np

class linear():
    def __init__(self, n: int, m: int):
        r = np.random.uniform(-1, 1, size=(n * m))
        self.A = r.reshape(n, m)

    def __call__(self, x):
        x = np.dot(x.T, self.A)
        return x.T

# Neural Network base class
# Defines the general structure of a neural network and all essential functionality
class net():
    def __init__(self, n: int, h: int, m: int):
        self.fc1 = linear(n, h)
        self.fc2 = linear(h, h)
        self.fc3 = linear(h, m)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

    # Uses stochastic gradient descent 
    def gradient(self, x, y):
        x

    # Applies results of loss 
    def backward(self, loss, lr):
        x

    @staticmethod
    def softmax(z):
        e = np.exp(z - np.max(z))
        sigma = e / e.sum()

        return sigma

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def least_squares(x, y):
        s = y - x
        return np.square(s).sum()

    @staticmethod
    def cross_entropy(p, q):
       return 

def test():
    print("Test ReLU...")
    x = np.array([-1, 0, 1])
    y = nn.relu(x)

    print(f"x = {x}")
    print(f"ReLU results = {y}")

    assert(np.array_equal(y, np.array([0, 0, 1])))
    print("ReLU is working")

    print("Testing softmax...")
    y = nn.softmax(x)
    print(f"Softmax results = {y}")

    print("Testing least squares...")
    y = np.array([-2, 1, 2])
    z = nn.least_squares(x, y)
    print(f"Least squares results = {z}")
    assert(z == 2)
    print("least squares is working")

if __name__ == "__main__":
    test()
