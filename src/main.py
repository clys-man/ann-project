import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lib.adaline import Adaline
from lib.mlp import MLP
from lib.perceptron import Perceptron
from lib.metrics import Metrics
from sklearn.preprocessing import MinMaxScaler
from lib.activation_functions import sigmoid, sigmoid_derivative

# Dados = pd.read_csv('data/data.csv')
# X = Dados.drop(['id', 'diagnosis', 'Unnamed: 32',], axis=1).values
# y = Dados['diagnosis']
# y = y.map({'M': 1, 'B': -1}).values

# models_data = {
#     'Perceptron': {'accuracy': [], 'sensitivity': [], 'specificity': []},
#     'Adaline': {'accuracy': [], 'sensitivity': [], 'specificity': []},
#     'MLP': {'accuracy': [], 'sensitivity': [], 'specificity': []},
# }
# perceptron_df = []
# for i in range(100):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = i)


#     perceptron = Perceptron(max_epochs=1000)
#     perceptron.fit(X_train, y_train)
#     y_pred = perceptron.predict(X_test)

#     metrics = Metrics(y_test, y_pred)
#     accuracy = metrics.accuracy()
#     sensitivity = metrics.sensitivity()
#     specificity = metrics.specificity()

#     models_data['Perceptron']['accuracy'].append(accuracy)
#     models_data['Perceptron']['sensitivity'].append(sensitivity)
#     models_data['Perceptron']['specificity'].append(specificity)

#     perceptron_df.append({'model': 'Perceptron', 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity})

#     adaline = Adaline(max_epochs=1000)
#     adaline.fit(X_train, y_train)
#     y_pred = adaline.predict(X_test)

#     metrics = Metrics(y_test, y_pred)
#     accuracy = metrics.accuracy()
#     sensitivity = metrics.sensitivity()
#     specificity = metrics.specificity()

#     models_data['Adaline']['accuracy'].append(accuracy)
#     models_data['Adaline']['sensitivity'].append(sensitivity)
#     models_data['Adaline']['specificity'].append(specificity)

#     perceptron_df.append({'model': 'Adaline', 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity})

#     learning_rate = 0.001
#     epocas = 1000
#     limiar = 0.6
#     camada_de_entrada = X_train.shape[1] # como dito em sala, isso é fixo.... mas poderia não ser.
#     camada_de_saida = 1 # no meu caso é uma classificação binária de atmosfera 'good' ou 'bad'
#     camadas_escondidas = [50, 20, 10]
#     camadas=[camada_de_entrada] + camadas_escondidas + [camada_de_saida]
#     erro = 0.005

#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.fit_transform(X_test)

#     mlp = MLP(hidden_layer_sizes=camadas, activation='tanh', max_iter=1000, learning_rate_init=learning_rate)
#     mlp.fit(X_train_scaled, y_train)
#     print(mlp.score(X_train_scaled, y_train), mlp.score(X_test_scaled, y_test))
#     y_pred = mlp.predict(X_test_scaled)

#     metrics = Metrics(y_test, y_pred)
#     accuracy = metrics.accuracy()
#     sensitivity = metrics.sensitivity()
#     specificity = metrics.specificity()

#     models_data['MLP']['accuracy'].append(accuracy)
#     models_data['MLP']['sensitivity'].append(sensitivity)
#     models_data['MLP']['specificity'].append(specificity)

#     perceptron_df.append({'model': 'MLP', 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity})


# for model, data in models_data.items():
#     print('Média - Mediana - Mínimo - Máximo - Desvio Padrão')
    
#     for metric, values in data.items():
#         print(f'{metric} & {np.mean(values):.2f} & {np.median(values):.2f} & {np.min(values):.2f} & {np.max(values):.2f} & {np.std(values):.2f} \\\\')
#     print()


# perceptron_df = pd.DataFrame(perceptron_df)
# sns.boxplot(data=perceptron_df, x='model', y='accuracy')
# plt.show()


Dados = np.genfromtxt("./data/aerogerador.dat", delimiter="\t", filling_values=np.nan)

Size = Dados.shape
X = Dados[:, 0].reshape(Size[0], 1)
y = Dados[:, 1].reshape(Size[0], 1)

models_data = {
    'Perceptron': {'mse': []},
    'Adaline': {'mse': []},
    'MLP': {'mse': []},
}
perceptron_df = []

perceptron = Perceptron(max_epochs=1000, activation_function='linear')
perceptron.fit(X, y)
y_pred_perceptron = perceptron.predict(X)
print('perceptron', y_pred_perceptron)

metrics = Metrics(y, y_pred_perceptron, True)
accuracy = metrics.mse()

models_data['Perceptron']['mse'].append(accuracy)

perceptron_df.append({'model': 'Perceptron', 'accuracy': accuracy, 'y_pred': y_pred_perceptron})

adaline = Adaline(max_epochs=1000, activation_function='linear')
adaline.fit(X, y)
y_pred_adaline = adaline.predict(X)

print('adaline', y_pred_adaline)

metrics = Metrics(y, y_pred_adaline, True)
accuracy = metrics.mse()

models_data['Adaline']['mse'].append(accuracy)

perceptron_df.append({'model': 'Adaline', 'accuracy': accuracy, 'y_pred': y_pred_adaline})

learning_rate = 0.001
epocas = 1000
limiar = 0.6
camada_de_entrada = X.shape[1] # como dito em sala, isso é fixo.... mas poderia não ser.
camada_de_saida = 1 # no meu caso é uma classificação binária de atmosfera 'good' ou 'bad'
camadas_escondidas = [50, 20, 10]
camadas=[camada_de_entrada] + camadas_escondidas + [camada_de_saida]
erro = 0.005

mlp = MLP(camadas=camadas, funcao_ativacao=sigmoide, funcao_ativacao_derivada=sigmoide_derivada, erro_minimo=erro, learning_rate=learning_rate, epocas=epocas, limiar = limiar)
mlp.fit(X, y)
y_pred_mlp = mlp.predict(X)


print('mlp', y_pred_mlp)

metrics = Metrics(y, y_pred_mlp, True)
accuracy = metrics.mse()

models_data['MLP']['mse'].append(accuracy)

perceptron_df.append({'model': 'MLP', 'accuracy': accuracy, 'y_pred': y_pred_mlp})

# best fit line
plt.scatter(X, y, color='green', label='Data points')
plt.plot(X, y_pred_perceptron, color='blue', label='Perceptron')
plt.plot(X, y_pred_adaline, color='red', label='Adaline')
plt.plot(X, y_pred_mlp, color='black', label='MLP')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True, ls='--', alpha=0.2, color='black')
plt.legend()
plt.show()

