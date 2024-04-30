import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from lib.adaline import Adaline
from lib.perceptron import Perceptron

learning_rate = 0.005
epocas = 100

Dados = pd.read_csv('data/spiral.csv').values

Size = Dados.shape
X = Dados[:, 0:Size[1]-1].reshape(Size[0], Size[1]-1)
y = Dados[:, Size[1]-1].reshape(Size[0], 1)

models_data = {
    'Perceptron': {'accuracy': [], 'sensitivity': [], 'specificity': []},
    'Adaline': {'accuracy': [], 'sensitivity': [], 'specificity': []}
}
perceptron_df = []
for i in range(100):
    perceptron = Perceptron(eta=learning_rate, max_epochs=epocas, loss_function='mse')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = i)
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    sensitivity = cm[1, 1] / np.sum(cm[1, :])
    specificity = cm[0, 0] / np.sum(cm[0, :])

    models_data['Perceptron']['accuracy'].append(accuracy)
    models_data['Perceptron']['sensitivity'].append(sensitivity)
    models_data['Perceptron']['specificity'].append(specificity)
    perceptron_df.append({
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'model': 'Perceptron'
    })

    adaline = Adaline(eta=learning_rate, max_epochs=epocas, epsilon=0.004, loss_function='mse')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = i)
    adaline.fit(X_train, y_train)

    y_pred = adaline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    sensitivity = cm[1, 1] / np.sum(cm[1, :])
    specificity = cm[0, 0] / np.sum(cm[0, :])

    models_data['Adaline']['accuracy'].append(accuracy)
    models_data['Adaline']['sensitivity'].append(sensitivity)
    models_data['Adaline']['specificity'].append(specificity)

    perceptron_df.append({
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'model': 'Adaline'
    })

for model, data in models_data.items():
    print('Média - Mediana - Mínimo - Máximo - Desvio Padrão')
    
    for metric, values in data.items():
        print(f'{metric} & {np.mean(values):.2f} & {np.median(values):.2f} & {np.min(values):.2f} & {np.max(values):.2f} & {np.std(values):.2f} \\\\')
    print()

perceptron_df = pd.DataFrame(perceptron_df)
sns.boxplot(data=perceptron_df, x='model', y='accuracy')
plt.show()