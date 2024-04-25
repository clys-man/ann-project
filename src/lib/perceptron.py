import numpy as np


class PerceptronSimples:
    def __init__(self, X_treino, Y_treino: np.ndarray, lr: float) -> None:
        self.X_treino = X_treino
        self.p, self.N = X_treino.shape
        self.Y_treino = Y_treino
        self.lr = lr
        self.X_treino = np.concatenate((-np.ones((1, self.N)), self.X_treino))

        self.w = np.zeros((self.p + 1, 1))
        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5

    def __sinal(self, u):
        if u >= 0:
            return 1
        else:
            return -1

    def treino(self):
        erro = True
        while erro:
            erro = False
            for t in range(self.N):
                x_t = self.X_treino[:, t].reshape(self.p + 1, 1)
                u_t = self.w.T @ x_t
                y_t = self.__sinal(u_t[0, 0])
                d_t = self.Y_treino[t, 0]
                e_t = d_t - y_t
                self.w = self.w + self.lr * (e_t / 2) * x_t
                if e_t != 0:
                    erro = True

    def teste(self, X_teste):
        N = X_teste.shape[1]
        X_teste = np.concatenate((-np.ones((1, N)), X_teste))
        Y_teste = np.zeros((N, 1))
        for t in range(N):
            x_t = X_teste[:, t].reshape(self.p + 1, 1)
            u_t = self.w.T @ x_t
            Y_teste[t, 0] = self.__sinal(u_t[0, 0])
        return Y_teste
