# Exercício 1: Iris Dataset (Setosa vs Versicolor)
# Objetivo: Classificar duas espécies de flores Iris que são linearmente separáveis
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time

# Adicionar pasta pai ao path para importar perceptron e util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perceptron import Perceptron
from util import plot_decision_regions

# PASSO 1: Carregar o Dataset Iris
print("=" * 60)
print("EXERCÍCIO 1: DATASET IRIS (SETOSA VS VERSICOLOR)")
print("=" * 60)

# Carregar dataset Iris - apenas classes 0 e 1 (linearmente separáveis)
iris = datasets.load_iris()
mask = iris.target != 2  # Remove classe 2 (Virginica)
X = iris.data[mask]
y = iris.target[mask]

# Usar apenas 2 features para visualização: sepal length [0] e petal length [2]
X = X[:, [0, 2]]

print(f"Dataset Iris (Setosa vs Versicolor):")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]} ({iris.feature_names[0]}, {iris.feature_names[2]})")
print(f"- Classes: {np.unique(y)} (0={iris.target_names[0]}, 1={iris.target_names[1]})")
print(f"- Distribuição das classes: {dict(zip(*np.unique(y, return_counts=True)))}")

# PASSO 2: Dividir em Treino e Teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"\nDivisão treino/teste (70/30):")
print(f"- Treino: {len(X_train)} amostras")
print(f"- Teste: {len(X_test)} amostras")

# PASSO 3: Normalização (StandardScaler)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

print(f"- Dados normalizados com StandardScaler")

# PASSO 4: Treinar o Perceptron
print(f"\nTreinando Perceptron...")
start_time = time.time()
ppn = Perceptron(learning_rate=0.01, n_epochs=50)
ppn.fit(X_train_std, y_train)
training_time = time.time() - start_time

# PASSO 5: Avaliar o Modelo
y_pred_train = ppn.predict(X_train_std)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = ppn.predict(X_test_std)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Dados combinados para visualização
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# PASSO 6: Resultados
print(f"\n" + "="*40)
print("RESULTADOS")
print("="*40)
print(f"- Acurácia no treinamento: {train_accuracy:.2%}")
print(f"- Acurácia no teste: {test_accuracy:.2%}")
print(f"- Tempo de treinamento: {training_time:.4f} segundos")
print(f"- Erros finais: {ppn.errors_history[-1]}")

# Verificar convergência
if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0) + 1
    print(f"- Convergiu na época: {conv_epoch}")
else:
    print(f"- Não convergiu completamente em {ppn.n_epochs} épocas")

# PASSO 7: Análise dos Pesos
print(f"\nPesos aprendidos:")
print(f"- w1 ({iris.feature_names[0]}): {ppn.weights[0]:.4f}")
print(f"- w2 ({iris.feature_names[2]}): {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")

# Equação da fronteira de decisão
if ppn.weights[1] != 0:
    slope = -ppn.weights[0]/ppn.weights[1]
    intercept = -ppn.bias/ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")

# PASSO 8: Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_test)
print(f"\nMatriz de Confusão (Teste):")
print(cm)
print(f"\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_test, target_names=iris.target_names[:2]))

# PASSO 9: Visualizações
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Regiões de Decisão
plt.subplot(2, 2, 1)
plot_decision_regions(X_combined_std, y_combined, classifier=ppn)
plt.title('Regiões de Decisão - Iris (Setosa vs Versicolor)')
plt.xlabel(f'{iris.feature_names[0]} (normalizada)')
plt.ylabel(f'{iris.feature_names[2]} (normalizada)')

# Subplot 2: Curva de Convergência
plt.subplot(2, 2, 2)
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, 
         marker='o', linewidth=2, markersize=6)
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência do Treinamento')
plt.grid(True, alpha=0.3)

# Subplot 3: Matriz de Confusão
plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names[:2],
            yticklabels=iris.target_names[:2])
plt.title('Matriz de Confusão')
plt.ylabel('Valor Real')
plt.xlabel('Predição')

# Subplot 4: Dataset Original (não normalizado)
plt.subplot(2, 2, 4)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', 
           label=iris.target_names[0], alpha=0.7, s=50)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', 
           label=iris.target_names[1], alpha=0.7, s=50)
plt.xlabel(f'{iris.feature_names[0]} (cm)')
plt.ylabel(f'{iris.feature_names[2]} (cm)')
plt.title('Dataset Original - Iris')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# PASSO 10: Análise e Conclusões
print(f"\n" + "="*60)
print("ANÁLISE E CONCLUSÕES")
print("="*60)

print("\n1. O Perceptron foi adequado para este problema?")
if test_accuracy >= 0.95:
    print(f"   ✓ SIM - Excelente desempenho ({test_accuracy:.2%} de acurácia)")
    print("   ✓ As classes Setosa e Versicolor são linearmente separáveis")
elif test_accuracy >= 0.80:
    print(f"   ≈ PARCIALMENTE - Bom desempenho ({test_accuracy:.2%})")
else:
    print(f"   ✗ NÃO - Baixo desempenho ({test_accuracy:.2%})")

print(f"\n2. Características observadas:")
print(f"   - Dataset linearmente separável: {'SIM' if 0 in ppn.errors_history else 'NÃO'}")
conv_epoch = ppn.errors_history.index(0) + 1 if 0 in ppn.errors_history else None
print(f"   - Convergência rápida: {'SIM' if conv_epoch and conv_epoch <= 10 else 'NÃO'} ({conv_epoch if conv_epoch else 'N/A'} épocas)")
print(f"   - Tempo de treinamento: {training_time:.4f}s (muito rápido)")

print(f"\n3. Comparação com expectativas:")
print(f"   - Acurácia esperada: ~100% (dados linearmente separáveis)")
print(f"   - Acurácia obtida: {test_accuracy:.2%}")
print(f"   - Resultado: {'Conforme esperado' if test_accuracy >= 0.95 else 'Abaixo do esperado'}")

print(f"\n4. Pergunta para reflexão - Versicolor vs Virginica:")
print(f"   Se usássemos classes 1 e 2 (Versicolor vs Virginica), o resultado seria")
print(f"   PIOR, pois essas classes não são linearmente separáveis com apenas")
print(f"   2 features. O Perceptron teria dificuldade para convergir.")

print(f"\n5. Melhorias sugeridas:")
print(f"   - Usar todas as 4 features (melhor representação)")
print(f"   - Implementar Perceptron multicamadas para 3 classes")
print(f"   - Comparar com SVM ou Random Forest")
print(f"   - Validação cruzada para resultados mais robustos")
