import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Metrics:
    def __init__(self, y, y_pred, regression = False):
        self.y = y
        self.y_pred = y_pred
        if not regression:
            self.cm = confusion_matrix(self.y, self.y_pred)

    def accuracy(self):
        return np.trace(self.cm) / np.sum(self.cm)
    
    def sensitivity(self):
        return self.cm[1, 1] / np.sum(self.cm[1, :])

    def specificity(self):
        return self.cm[0, 0] / np.sum(self.cm[0, :])
    
    def mse(self):
        return np.mean((self.y_pred - self.y) ** 2)