# Exercício 2: Moons Dataset
# Objetivo: Demonstrar as limitações do perceptron com dados não-linearmente separáveis
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time

# Adicionar pasta pai ao path para importar perceptron e util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perceptron import Perceptron
from util import plot_decision_regions

# PASSO 1: Gerar o Dataset Moons
print("=" * 60)
print("EXERCÍCIO 2: DATASET MOONS (NÃO-LINEARMENTE SEPARÁVEL)")
print("=" * 60)

# Gerar dataset moons - duas luas entrelaçadas
X, y = make_moons(
    n_samples=200,
    noise=0.15,  # Adiciona ruído realista
    random_state=42
)

print(f"Dataset Moons:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]}")
print(f"- Classes: {np.unique(y)} (0=Lua inferior, 1=Lua superior)")
print(f"- Distribuição das classes: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"- Ruído adicionado: 15%")
print(f"- Tipo: NÃO-LINEARMENTE SEPARÁVEL")

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
ppn = Perceptron(learning_rate=0.01, n_epochs=100)  # Mais épocas pois não vai convergir
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
    print(f"- NÃO CONVERGIU em {ppn.n_epochs} épocas")
    min_errors = min(ppn.errors_history)
    min_epoch = ppn.errors_history.index(min_errors) + 1
    print(f"- Menor número de erros: {min_errors} (época {min_epoch})")

# PASSO 7: Análise dos Pesos
print(f"\nPesos aprendidos:")
print(f"- w1 (Feature 1): {ppn.weights[0]:.4f}")
print(f"- w2 (Feature 2): {ppn.weights[1]:.4f}")
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
print(classification_report(y_test, y_pred_test, target_names=['Lua Inferior', 'Lua Superior']))

# PASSO 9: Visualizações
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Regiões de Decisão
plt.subplot(2, 2, 1)
plot_decision_regions(X_combined_std, y_combined, classifier=ppn)
plt.title('Regiões de Decisão - Moons\n(Linha reta não consegue separar as luas)')
plt.xlabel('Feature 1 (normalizada)')
plt.ylabel('Feature 2 (normalizada)')

# Subplot 2: Curva de Convergência
plt.subplot(2, 2, 2)
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, 
         marker='o', linewidth=2, markersize=4, color='red')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência do Treinamento\n(Erros nunca chegam a zero)')
plt.grid(True, alpha=0.3)

# Subplot 3: Matriz de Confusão
plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Lua Inferior', 'Lua Superior'],
            yticklabels=['Lua Inferior', 'Lua Superior'])
plt.title('Matriz de Confusão')
plt.ylabel('Valor Real')
plt.xlabel('Predição')

# Subplot 4: Dataset Original (não normalizado)
plt.subplot(2, 2, 4)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', 
           label='Lua Inferior', alpha=0.7, s=50)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', 
           label='Lua Superior', alpha=0.7, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dataset Original - Moons\n(Formato de luas entrelaçadas)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# PASSO 10: Análise e Conclusões
print(f"\n" + "="*60)
print("ANÁLISE E CONCLUSÕES")
print("="*60)

print("\n1. O Perceptron foi adequado para este problema?")
if test_accuracy <= 0.60:
    print(f"   ✗ NÃO - Baixo desempenho ({test_accuracy:.2%} de acurácia)")
    print("   ✗ Dados não-linearmente separáveis")
elif test_accuracy <= 0.80:
    print(f"   ≈ PARCIALMENTE - Desempenho limitado ({test_accuracy:.2%})")
else:
    print(f"   ? INESPERADO - Desempenho superior ao esperado ({test_accuracy:.2%})")

print(f"\n2. Limitações observadas:")
print(f"   - Dataset NÃO-linearmente separável: fronteira curva necessária")
print(f"   - Perceptron usa apenas linha reta: inadequado para este formato")
print(f"   - Convergência: {'IMPOSSÍVEL' if 0 not in ppn.errors_history else 'INESPERADA'}")
print(f"   - Acurácia próxima ao acaso: {'SIM' if test_accuracy < 0.65 else 'NÃO'}")

print(f"\n3. Comparação com expectativas:")
print(f"   - Acurácia esperada: ~50-60% (próximo ao acaso)")
print(f"   - Acurácia obtida: {test_accuracy:.2%}")
print(f"   - Resultado: {'Conforme esperado' if 0.5 <= test_accuracy <= 0.65 else 'Diferente do esperado'}")

print(f"\n4. Por que o Perceptron falha aqui?")
print(f"   - As duas luas são entrelaçadas em formato curvo")
print(f"   - Uma linha reta NUNCA pode separar perfeitamente essas formas")
print(f"   - É necessária uma fronteira de decisão não-linear")
print(f"   - O Perceptron simples só consegue aprender funções lineares")

print(f"\n5. Soluções possíveis:")
print(f"   - Perceptron Multicamadas (MLP) com camadas ocultas")
print(f"   - Support Vector Machine (SVM) com kernel RBF")
print(f"   - Árvores de Decisão ou Random Forest")
print(f"   - Redes Neurais profundas")
print(f"   - Transformação de features para espaço dimensional maior")

print(f"\n6. Lição aprendida:")
print(f"   O Perceptron é um algoritmo LINEAR e tem limitações fundamentais.")
print(f"   Para problemas não-lineares, algoritmos mais sofisticados são necessários.")