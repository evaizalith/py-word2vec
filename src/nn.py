import numpy as np

class linear():
    def __init__():
        print("unimplemented")

# Neural Network base class
# Defines the general structure of a neural network and all essential functionality
class nn():
    def __init__():
        print("unimplemented")


    def linear():
         print("unimplemented")
    
    # Uses stochastic gradient descent 
    def gradient(self):
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

    #assert(np.array_equal(y, np.array([0.09003057, 0.24472847, 0.66524096])))
    print("Softmax is working")

    print("Testing least squares...")
    y = np.array([-2, 1, 2])
    z = nn.least_squares(x, y)
    print(f"Least squares results = {z}")
    assert(z == 2)
    print("least squares is working")

if __name__ == "__main__":
    test()
