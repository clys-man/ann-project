import numpy as np
from lib.activation_functions import signal, linear, sigmoid

class Model:
    def _handleActivationFunction(self, x):
        if self.activation_function == 'signal':
            return signal(x)
        elif self.activation_function == 'linear':
            return linear(x)
        elif self.activation_function == 'sigmoid':
            return sigmoid(x)
        else:
            raise ValueError('Função de ativação não reconhecida')
        
    def _mse(self, X, y, W):
        y_pred = W.T @ X
        return np.mean((y - y_pred) ** 2)