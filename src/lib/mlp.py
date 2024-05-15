import numpy as np

class MLP:
    def __init__(self, camadas, funcao_ativacao, funcao_ativacao_derivada, erro_minimo, learning_rate=0.05, epocas=10000, limiar=0.70):
        self.camadas = camadas
        self.funcao_ativacao = funcao_ativacao
        self.funcao_ativacao_derivada = funcao_ativacao_derivada
        self.learning_rate = learning_rate
        self.epocas = epocas
        self.limiar = limiar
        self.erro_minimo = erro_minimo
        self.pesos, self.biases = self.iniciando_pesos()
        

    def iniciando_pesos(self):
        np.random.seed(42)  
        pesos = []  
        biases = []  

        for i in range(len(self.camadas) - 1):
            
            pesos.append(np.random.randn(self.camadas[i], self.camadas[i + 1]))
            biases.append(np.zeros((1, self.camadas[i + 1])))

        return pesos, biases

    def treinar(self, X_train, y_train):
        m = X_train.shape[0] 
        epoca = 0

        for epoca in range(self.epocas):
            saidas = [X_train] 
            zs = [] 

            for i in range(len(self.camadas) - 1):
                z = saidas[-1] @ self.pesos[i] + self.biases[i]
                zs.append(z) 
                
                a = self.funcao_ativacao(z)
                saidas.append(a) 

            erro = np.sum(np.square(y_train.reshape(-1, 1) - saidas[-1])) / (2 * m)
            if (erro <= self.erro_minimo):
                #print('parou devido ao erro')
                break
            
            delta = (saidas[-1] - y_train.reshape(-1, 1)) * self.funcao_ativacao_derivada(zs[-1])
            dW = (1 / m) * (saidas[-2].T @ delta)  
            db = (1 / m) * np.sum(delta, axis=0, keepdims=True)  
            peso_atualizado = [dW]  
            bias_atualizado = [db]
        

            for i in range(len(self.camadas) - 3, -1, -1):
                delta = (delta @ self.pesos[i + 1].T) * self.funcao_ativacao_derivada(zs[i])
                
                dW = (1 / m) * (saidas[i].T @ delta)
                
                db = (1 / m) * np.sum(delta, axis=0, keepdims=True)
                
                peso_atualizado.insert(0, dW)
                bias_atualizado.insert(0, db)

            for i in range(len(self.pesos)):
                self.pesos[i] -= self.learning_rate * peso_atualizado[i]
                self.biases[i] -= self.learning_rate * bias_atualizado[i]
                
            epoca+=1   
            if(epoca == self.epocas):    
                #print('parou devido ao numero de max de épocas pré definido')
                break  
        

    def prever(self, X):
        saidas = [X]
        for i in range(len(self.camadas) - 1):
            z = saidas[-1] @ self.pesos[i] + self.biases[i]
            a = self.funcao_ativacao(z)
            saidas.append(a)

        # mudar o limite do limiar dependendo da função de ativação
        y_pred = np.where(saidas[-1] >= self.limiar, 1, -1)
        return y_pred

    def acuracia(self, y_pred, y_true):
        return np.mean(y_pred == y_true)

