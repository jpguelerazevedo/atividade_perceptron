import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):    
    # PASSO 1: Configurar cores e marcadores
    markers = ('o', 's') # círculo para classe 0, quadrado para classe 1
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # PASSO 2: Criar grade de pontos (meshgrid)
    # Definir limites do gráfico com margem
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Criar arrays de coordenadas
    # xx1: matriz com coordenadas x repetidas em cada linha
    # xx2: matriz com coordenadas y repetidas em cada coluna
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
    # PASSO 3: Classificar cada ponto da grade
    # Achatar as matrizes e criar pares (x1, x2)
    grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
    # Prever a classe de cada ponto
    Z = classifier.predict(grid_points)
    # Reformatar para o shape da grade
    Z = Z.reshape(xx1.shape)
    # PASSO 4: Plotar regiões coloridas
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # PASSO 5: Plotar pontos de treinamento
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
        x=X[y == cl, 0], # coordenadas x dos pontos da classe cl
        y=X[y == cl, 1], # coordenadas y dos pontos da classe cl
        alpha=0.8,
        c=colors[idx],
        marker=markers[idx],
        label=f'Classe {cl}',
        edgecolor='black')
    plt.legend(loc='upper left')