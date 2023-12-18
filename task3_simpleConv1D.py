from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import math


# [Problem 1] Creating a one-dimensional convolutional layer class that limits the number of channels to one
class Conv1D:

    def forward_propagate(self, x, w, b):
        # Output value (which will converted to np.array)
        a = []
        for i in range(len(w) - 1):
            a.append(np.matmul(x[i:i + len(w)], w) + b[0])

        return np.array(a)

    def backward_propagate(self, x, w, da):
        """
        The backward propagate intends to find three factors:
        dw, db and dx

        dw, db:
          Using the formula similar to DNN model

        dx:
          Using the weight value multiply with the derivative of activate function da

        """
        # Calculate db
        db = np.sum(da)

        # Calculate dw
        dw = []
        for i in range(len(w)):
            dw.append(np.matmul(da, x[i:i + len(da)]))
        dw = np.array(dw)

        # Calculate dx
        dx = []
        # Adding all the shared errors
        # The errors lies in the two heads of array (j - s < 0) and (j - s > N - 1)
        new_w = np.insert(w[::-1], 0, 0)  # Reverse the weight array
        new_w = np.append(new_w, 0)
        for i in range(len(new_w) - 1):
            dx.append(np.matmul(da, new_w[i:i + len(da)]))
        dx = np.array(dx[::-1])  # Reverse again

        return dw, db, dx

# [Problem 2] Output size calculation after one-dimensional convolution

def output_size_calculation(n_in, F, P=0, S=1):
    n_out = int((n_in + 2 * P - F) / S + 1)
    return n_out

# [Problem 3] Experiment of one-dimensional convolutional layer with small array
x = np.array([1, 2, 3, 4])
w = np.array([3, 5, 7])
b = np.array([1])
da = np.array([10, 20])


Conv1D_model = Conv1D()
dw, db, dx = Conv1D_model.backward_propagate(x, w, da)

print("Forward: ", Conv1D_model.forward_propagate(x, w, b))
print("dw: ", dw)
print("db: ", db)
print("dx: ", dx)


#[Problem 4] Creating a one-dimensional convolutional layer class that does not limit the number of channels
x = np.ones((28, 28))
y = np.pad(x, pad_width=((0,0), (2,0)))


class Conv1DFull:

    def __init__(self, filter_size, initializer, optimizer, channels_in=1, channels_out=1, pad=0):
        self.filter_size = filter_size
        self.optimizier = optimizer
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_out = None
        self.pad = pad
        self.W = initializer.W(channels_out, channels_in, filter_size)
        self.B = initializer.B(channels_out)

    def forward_propagate(self, X):
        self.n_in = X.shape[-1]
        self.n_out = output_size_calculation(self.n_in, self.filter_size, self.pad)

        X = X.reshape(self.channels_in, self.n_in)
        self.X = np.pad(X, ((0, 0), ((self.filter_size - 1), 0)))
        self.X1 = np.zeros((self.channels_in, self.filter_size, self.n_in + (self.filter_size - 1)))

        for i in range(self.filter_size):
            self.X1[:, i] = np.roll(self.X, -i, axis=1)

        A = np.sum(self.X1[:, :, self.filter_size - 1 - self.pad:self.n_in + self.pad] * self.W[:, :, :, np.newaxis],
                   axis=(1, 2)) + self.B.reshape(-1, 1)

        return A

    def backward_propagate(self, dA):

        self.dW = np.sum(np.dot(dA, self.X1[:, :, self.filter_size - 1 - self.pad:self.n_in + self.pad, np.newaxis]),
                         axis=-1)
        self.dB = np.sum(dA, axis=1)
        self.dA = np.pad(dA, ((0, 0), (0, (self.filter_size - 1))))
        self.dA1 = np.zeros((self.channels_out, self.filter_size, self.dA.shape[-1]))

        for i in range(self.filter_size):
            self.dA1[:, i] = np.roll(self.dA, i, axis=1)

        dX = np.sum(np.matmul(self.W, self.dA1), axis=0)
        self.optimizer.update(self)

        return dX

class WeightInitializer:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def initialize_weights(self, shape):
        std_dev = np.sqrt(self.gamma / np.prod(shape[:-1]))
        return np.random.normal(loc=0, scale=std_dev, size=shape)

    def initialize_biases(self, shape):
        return np.zeros(shape)

    def W(self, channels_out, channels_in, filter_size):
        # Initialize weights and biases separately
        self.W = self.initialize_weights((channels_out, channels_in, filter_size))

    def B(self, channels_out):
        self.B = self.initialize_biases((channels_out,))


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB
        return

conv_model = Conv1DFull(filter_size=3, initializer=WeightInitializer(0.01),
                        optimizer=SGD(0.01), channels_in=2, channels_out=3, pad=0)