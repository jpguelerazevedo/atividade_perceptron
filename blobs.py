# blobs.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from util import plot_decision_regions

# PASSO 1: Gerar o Dataset
print("=" * 50)
print("EXEMPLO: BLOBS SINTÉTICOS")
print("=" * 50)

# make_blobs cria clusters gaussianos
X, y = datasets.make_blobs(
    n_samples=200, # Total de pontos
    n_features=2, # Número de features (2 para visualização)
    centers=2, # Número de clusters (classes)
    cluster_std=1.5, # Desvio padrão dos clusters
    center_box=(-5, 5), # Limites para os centros
    random_state=42 # Seed para reprodutibilidade
)

print(f"Dataset gerado:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]}")
print(f"- Classes: {np.unique(y)}")

# PASSO 2: Dividir em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # 30% para teste
    random_state=42,
    stratify=y # Mantém proporção das classes
)

print(f"\nDivisão treino/teste:")
print(f"- Treino: {len(X_train)} amostras")
print(f"- Teste: {len(X_test)} amostras")

# PASSO 3: Normalização (Importante!)
"""
Por que normalizar?
- Garante que todas features tenham a mesma escala
- Previne que features com valores grandes dominem
- Acelera convergência
- Método: z-score (média=0, desvio=1)
"""
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train) # Fit no treino
X_test_std = scaler.transform(X_test) # Apenas transform no teste

# PASSO 4: Treinar o Perceptron
ppn = Perceptron(learning_rate=0.01, n_epochs=50)
ppn.fit(X_train_std, y_train)

# PASSO 5: Avaliar o Modelo
# Acurácia no treino
y_pred_train = ppn.predict(X_train_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
# Acurácia no teste
y_pred_test = ppn.predict(X_test_std)
test_accuracy = accuracy_score(y_test, y_pred_test)
# Acurácia total
y_pred_total = np.concatenate([y_pred_train, y_pred_test])
y_total = np.concatenate([y_train, y_test])
total_accuracy = accuracy_score(y_total, y_pred_total)
# Total de amostras classificadas erradas (treino + teste)
train_errors = np.sum(y_pred_train != y_train)
test_errors = np.sum(y_pred_test != y_test)
total_errors = int(train_errors + test_errors)
print(f"- Total de amostras classificadas erradas (treino + teste): {total_errors}")

print(f"\nResultados:")
print(f"- Acurácia no conjunto de treinamento: {train_accuracy:.2%}")
print(f"- Acurácia no conjunto de teste: {test_accuracy:.2%}")
print(f"- Acurácia em todo o conjunto: {total_accuracy:.2%}")
print(f"- Erros finais no treino: {ppn.errors_history[-1]}")

# Verificar convergência
if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0)
    print(f"- Convergiu na época: {conv_epoch + 1}")
else:
    print("- Não convergiu completamente")
    
# PASSO 6: Análise dos Pesos Aprendidos
print(f"\nPesos aprendidos:")
print(f"- w1: {ppn.weights[0]:.4f}")
print(f"- w2: {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")

# A equação da fronteira de decisão é:
# w1*x1 + w2*x2 + bias = 0
# ou seja: x2 = -(w1/w2)*x1 - (bias/w2)
if ppn.weights[1] != 0:
    slope = -ppn.weights[0]/ppn.weights[1]
    intercept = -ppn.bias/ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")

# PASSO 7: Visualizar Resultados
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Regiões de Decisão
plt_decision = plt.subplot(1,2,1)
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt_decision.set_title('Regiões de Decisão - Blobs')
plt_decision.set_xlabel('Feature 1 (normalizada)')
plt_decision.set_ylabel('Feature 2 (normalizada)')
plt_decision.legend(loc='upper left')

# Subplot 2: Curva de Convergência
plt_convergence = plt.subplot(1,2,2)
plt_convergence.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o')
plt_convergence.set_xlabel('Épocas')
plt_convergence.set_ylabel('Número de erros')
plt_convergence.set_title('Convergência do Treinamento')
plt_convergence.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()