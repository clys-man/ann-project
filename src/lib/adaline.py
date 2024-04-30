import numpy as np

class Adaline:
    def __init__(self, eta, epsilon, max_epochs):
        self.eta = eta
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.W = None

    def __sinal(self, u):
        if u >= 0:
            return 1
        else:
            return -1

    def __eqm(self, x, y, W):
        return np.mean((y - W.T @ x) ** 2)

    def fit(self, X_train, y_train):
        X_train = X_train.T
        X_train = np.concatenate((-np.ones((1, X_train.shape[1])), X_train), axis=0)
        self.W = np.random.random_sample((X_train.shape[0], 1)) - 0.5
        self.p, self.N = X_train.shape

        epoch = 0
        while epoch <= self.max_epochs:
            eqm1 = self.__eqm(X_train, y_train, self.W)

            for t in range(self.N):
                x_t = X_train[:, t].reshape(X_train.shape[0], 1)
                u_t = self.W.T @ x_t
                y_t = self.__sinal(u_t[0, 0])
                d_t = y_train[t, 0]
                e_t = d_t - y_t
                self.W = self.W + self.eta * e_t * x_t

            epoch += 1
            eqm2 = self.__eqm(X_train, y_train, self.W)

            if (eqm2 - eqm1)  <= self.epsilon:
                break

    def predict(self, X_test):
        X_test = X_test.T
        X_test = np.concatenate((-np.ones((1, X_test.shape[1])), X_test), axis=0)
        y_pred = np.zeros((X_test.shape[1], 1))

        for t in range(X_test.shape[1]):
            x_t = X_test[:, t].reshape(X_test.shape[0], 1)
            u_t = self.W.T @ x_t
            y_t = self.__sinal(u_t[0, 0])
            y_pred.itemset(t, y_t)

        return y_pred