import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

class GetMiniBatch:
    """
    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      訓練データ
    y : 次の形のndarray, shape (n_samples, 1)
      正解値
    batch_size : int
      バッチサイズ
    seed : int
      NumPyの乱数のシード
    """
    def __init__(self, X, y, batch_size = 20, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self._X = X[shuffle_index]
        self._y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(int)

    def __len__(self):
        return self._stop

    def __getitem__(self,item):
        p0 = item*self.batch_size
        p1 = item*self.batch_size + self.batch_size
        return self._X[p0:p1], self._y[p0:p1]

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter*self.batch_size
        p1 = self._counter*self.batch_size + self.batch_size
        self._counter += 1
        return self._X[p0:p1], self._y[p0:p1]


class ScratchSimpleNeuralNetrowkClassifier():
    def __init__(self,
                 batch_size=20,
                 n_features=784,
                 n_nodes1=400,
                 n_nodes2=200,
                 n_output=10,
                 sigma=0.005,
                 lr=0.01,
                 epoch=5, verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.n_output = n_output
        self.sigma = sigma
        self.lr = lr
        self.epoch = epoch
        self.loss_train = []
        self.loss_val = []

    def fit(self, X, y, X_val=None, y_val=None):
        (self.W1, self.W2, self.W3,
         self.B1, self.B2, self.B3) = self.parameter_initialize()
        for _ in tqdm(range(self.epoch)):
            get_mini_batch = GetMiniBatch(X, y, batch_size=self.batch_size)
            for mini_X_train, mini_y_train in get_mini_batch:
                self.forward(mini_X_train)
                self.backward(mini_X_train, mini_y_train)
            self.forward(X)
            self.loss_train.append(self.cross_entropy_error(y, self.Z3))
            if X_val is not None:
                self.forward(X_val)
                self.loss_val.append(self.cross_entropy_error(y_val, self.Z3))
        if self.verbose:
            if X_val is None:
                print(self.loss_train)
            else:
                print(self.loss_train, self.loss_val)

    def parameter_initialize(self):
        W1 = self.sigma * np.random.randn(self.n_features, self.n_nodes1)
        W2 = self.sigma * np.random.randn(self.n_nodes1, self.n_nodes2)
        W3 = self.sigma * np.random.randn(self.n_nodes2, self.n_output)
        B1 = self.sigma * np.random.randn(1, self.n_nodes1)
        B2 = self.sigma * np.random.randn(1, self.n_nodes2)
        B3 = self.sigma * np.random.randn(1, self.n_output)
        return W1, W2, W3, B1, B2, B3

    def forward(self, X):
        self.A1 = X @ self.W1 + self.B1
        self.Z1 = self.tanh_function(self.A1)
        self.A2 = self.Z1 @ self.W2 + self.B2
        self.Z2 = self.tanh_function(self.A2)
        self.A3 = self.Z2 @ self.W3 + self.B3
        self.Z3 = self.softmax(self.A3)

    def backward(self, X, y):
        dA3 = (self.Z3 - y) / self.batch_size
        dW3 = self.Z2.T @ dA3
        dB3 = np.sum(dA3, axis=0)
        dZ2 = dA3 @ self.W3.T
        dA2 = dZ2 * (1 - self.tanh_function(self.A2) ** 2)
        dW2 = self.Z1.T @ dA2
        dB2 = np.sum(dA2, axis=0)
        dZ1 = dA2 @ self.W2.T
        dA1 = dZ1 * (1 - self.tanh_function(self.A1) ** 2)
        dW1 = X.T @ dA1
        dB1 = np.sum(dA1, axis=0)
        self.W3 -= self.lr * dW3
        self.B3 -= self.lr * dB3
        self.W2 -= self.lr * dW2
        self.B2 -= self.lr * dB2
        self.W1 -= self.lr * dW1
        self.B1 -= self.lr * dB1

    def tanh_function(self, X):
        result = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
        return result

    def softmax(self, X):
        result = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
        return result

    def cross_entropy_error(self, y, Z):
        L = - np.sum(y * np.log(Z + 1e-7)) / len(y)
        return L

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.Z3, axis=1)

#### Running Sctratch ###
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

X_train = X_train[:500, :]
X_test = X_test[:500, :]
y_train = y_train[:500]
y_test = y_test[:100]

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train_one_hot = enc.fit_transform(y_train[:, np.newaxis])
y_test_one_hot = enc.transform(y_test[:, np.newaxis])

model_nn = ScratchSimpleNeuralNetrowkClassifier(batch_size=4)

model_nn.fit(X_train, y_train_one_hot, X_test, y_test_one_hot)
pred = model_nn.predict(X_test[0, :])

#visualize


