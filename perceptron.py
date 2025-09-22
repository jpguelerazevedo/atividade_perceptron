# perceptron.py
import numpy as np

class Perceptron:    
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_history = []

    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):        
        n_samples, n_features = X.shape
        # PASSO 1: Inicialização dos pesos
        # Começamos com pesos zero (também comum: pequenos valores aleatórios)
        self.weights = np.zeros(n_features)
        self.bias = 0
        # PASSO 2: Loop de treinamento
        for epoch in range(self.n_epochs):
            errors = 0
            # PASSO 3: Para cada exemplo de treinamento
            for idx, x_i in enumerate(X):
                # 3.1: Calcula a saída líquida (net input)
                # net = w1*x1 + w2*x2 + ... + wn*xn + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                # 3.2: Aplica função de ativação
                y_predicted = self.activation(linear_output)
                # 3.3: Calcula o erro
                error = y[idx] - y_predicted
                # 3.4: Atualiza pesos e bias (Regra Delta)
                # Se error = 0: não há atualização
                # Se error = 1: move fronteira para incluir ponto
                # Se error = -1: move fronteira para excluir ponto
                update = self.learning_rate * error
                self.weights += update * x_i
                self.bias += update
                # Conta erros para monitoramento
                errors += int(update != 0.0)
            self.errors_history.append(errors)
            # Parada antecipada se convergiu
            if errors == 0:
                print(f"Convergiu na época {epoch + 1}")
                break

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return self.activation(self.net_input(X))