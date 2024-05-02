import numpy as np
from lib.model import Model

class Perceptron(Model):
    def __init__(self, eta=0.001, max_epochs=1000, activation_function='signal'):
        self.eta = eta
        self.max_epochs = max_epochs
        self.W = None
        self.activation_function = activation_function

    def fit(self, X_train, y_train):
        X_train = X_train.T
        X_train = np.concatenate((-np.ones((1, X_train.shape[1])), X_train), axis=0)
        y_train.shape = (len(y_train), 1)
        self.W = np.random.random_sample((X_train.shape[0], 1)) - 0.5
        epoch = 0

        while True:
            epoch += 1
            for t in range(X_train.shape[1]):
                x_t = X_train[:, t].reshape(X_train.shape[0], 1)
                u_t = self.W.T @ x_t
                y_t = self._handleActivationFunction(u_t[0, 0])
                d_t = y_train[t, 0]
                error = d_t - y_t
                self.W += self.eta * error * x_t

            if epoch >= self.max_epochs:
                break

    def predict(self, X_test):
        X_test = X_test.T
        X_test = np.concatenate((-np.ones((1, X_test.shape[1])), X_test), axis=0)

        y_pred = np.zeros((X_test.shape[1], 1))
        for t in range(X_test.shape[1]):
            x_t = X_test[:, t].reshape(X_test.shape[0], 1)
            u_t = self.W.T @ x_t
            y_t = self._handleActivationFunction(u_t[0, 0])
            y_pred.itemset(t, y_t)
        
        return y_pred

    def acuracia(self, y, y_pred):
        y.shape = (len(y), 1)
        y_pred.shape = (len(y_pred), 1)
        acuracia = np.sum(y == y_pred) / len(y)

        return acuracia
