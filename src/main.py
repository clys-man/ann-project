import numpy as np
from lib.adaline import Adaline
from lib.perceptron import PerceptronSimples

x = np.array(
    [
        [1, 1],
        [0, 1],
        [0, 2],
        [1, 0],
        [2, 2],
        [4, 1.5],
        [1.5, 6],
        [3, 5],
        [3, 3],
        [6, 4],
    ]
)

y = np.array(
    [
        [1],
        [1],
        [1],
        [1],
        [1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
    ]
)

perceptron = PerceptronSimples(x.T, y, 0.1)
perceptron.treino()

X_teste = np.array([[6, 4]])
Y_teste = perceptron.teste(X_teste.T)

print(Y_teste)


adaline = Adaline(x.T, y, 0.1, 100, 0.01)
adaline.treino()
Y_teste = adaline.teste(X_teste.T)
print(Y_teste)
