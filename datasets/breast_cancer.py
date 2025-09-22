# Exercício 3: Breast Cancer Wisconsin
# Objetivo: Aplicar perceptron em um problema médico real
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time

# Adicionar pasta pai ao path para importar perceptron e util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perceptron import Perceptron
from util import plot_decision_regions

# PASSO 1: Carregar o Dataset Breast Cancer
print("=" * 60)
print("EXERCÍCIO 3: DATASET BREAST CANCER WISCONSIN")
print("=" * 60)

# Carregar dataset breast cancer
cancer = load_breast_cancer()
X_full = cancer.data
y = cancer.target

print(f"Dataset Breast Cancer Wisconsin:")
print(f"- Amostras: {X_full.shape[0]}")
print(f"- Features: {X_full.shape[1]}")
print(f"- Classes: {cancer.target_names} (0=Malignant, 1=Benign)")
print(f"- Distribuição das classes: {dict(zip(cancer.target_names, np.bincount(y)))}")

print(f"\nPrimeiras features disponíveis:")
for i, name in enumerate(cancer.feature_names[:10]):
    print(f"  [{i}] {name}")
print(f"  ... e mais {len(cancer.feature_names)-10} features")

# VERSÃO A: Apenas 2 features para visualização
print(f"\n" + "="*50)
print("VERSÃO A: DUAS FEATURES PARA VISUALIZAÇÃO")
print("="*50)

# Usar as duas features mais discriminativas: mean area [5] e worst concave points [27]
X_2d = X_full[:, [5, 27]]  # mean area, worst concave points
feature_names_2d = [cancer.feature_names[5], cancer.feature_names[27]]

print(f"Features selecionadas:")
print(f"- {feature_names_2d[0]}")
print(f"- {feature_names_2d[1]}")

# Dividir em treino/teste
X_train_2d, X_test_2d, y_train, y_test = train_test_split(
    X_2d, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Normalizar
scaler_2d = StandardScaler()
X_train_2d_std = scaler_2d.fit_transform(X_train_2d)
X_test_2d_std = scaler_2d.transform(X_test_2d)

# Treinar Perceptron 2D
print(f"\nTreinando Perceptron (2D)...")
start_time = time.time()
ppn_2d = Perceptron(learning_rate=0.01, n_epochs=100)
ppn_2d.fit(X_train_2d_std, y_train)
training_time_2d = time.time() - start_time

# Avaliar 2D
y_pred_train_2d = ppn_2d.predict(X_train_2d_std)
train_accuracy_2d = accuracy_score(y_train, y_pred_train_2d)

y_pred_test_2d = ppn_2d.predict(X_test_2d_std)
test_accuracy_2d = accuracy_score(y_test, y_pred_test_2d)

print(f"\nResultados (2D):")
print(f"- Acurácia no treinamento: {train_accuracy_2d:.2%}")
print(f"- Acurácia no teste: {test_accuracy_2d:.2%}")
print(f"- Tempo de treinamento: {training_time_2d:.4f} segundos")
print(f"- Erros finais: {ppn_2d.errors_history[-1]}")

# Convergência 2D
if 0 in ppn_2d.errors_history:
    conv_epoch_2d = ppn_2d.errors_history.index(0) + 1
    print(f"- Convergiu na época: {conv_epoch_2d}")
else:
    print(f"- Não convergiu em {ppn_2d.n_epochs} épocas")

# VERSÃO B: Todas as 30 features
print(f"\n" + "="*50)
print("VERSÃO B: TODAS AS 30 FEATURES")
print("="*50)

# Dividir treino/teste para versão completa
X_train_full, X_test_full, _, _ = train_test_split(
    X_full, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Normalizar todas as features
scaler_full = StandardScaler()
X_train_full_std = scaler_full.fit_transform(X_train_full)
X_test_full_std = scaler_full.transform(X_test_full)

# Treinar Perceptron completo
print(f"Treinando Perceptron (30D)...")
start_time = time.time()
ppn_full = Perceptron(learning_rate=0.01, n_epochs=100)
ppn_full.fit(X_train_full_std, y_train)
training_time_full = time.time() - start_time

# Avaliar versão completa
y_pred_train_full = ppn_full.predict(X_train_full_std)
train_accuracy_full = accuracy_score(y_train, y_pred_train_full)

y_pred_test_full = ppn_full.predict(X_test_full_std)
test_accuracy_full = accuracy_score(y_test, y_pred_test_full)

print(f"\nResultados (30D):")
print(f"- Acurácia no treinamento: {train_accuracy_full:.2%}")
print(f"- Acurácia no teste: {test_accuracy_full:.2%}")
print(f"- Tempo de treinamento: {training_time_full:.4f} segundos")
print(f"- Erros finais: {ppn_full.errors_history[-1]}")

# Convergência completa
if 0 in ppn_full.errors_history:
    conv_epoch_full = ppn_full.errors_history.index(0) + 1
    print(f"- Convergiu na época: {conv_epoch_full}")
else:
    print(f"- Não convergiu em {ppn_full.n_epochs} épocas")

# MÉTRICAS MÉDICAS DETALHADAS
print(f"\n" + "="*50)
print("ANÁLISE MÉDICA DETALHADA")
print("="*50)

# Matriz de confusão 2D
cm_2d = confusion_matrix(y_test, y_pred_test_2d)
print(f"\nMatriz de Confusão (2D):")
print(cm_2d)

# Matriz de confusão completa
cm_full = confusion_matrix(y_test, y_pred_test_full)
print(f"\nMatriz de Confusão (30D):")
print(cm_full)

# Relatórios de classificação
print(f"\nRelatório de Classificação (2D):")
print(classification_report(y_test, y_pred_test_2d, target_names=cancer.target_names))

print(f"\nRelatório de Classificação (30D):")
print(classification_report(y_test, y_pred_test_full, target_names=cancer.target_names))

# Análise médica específica
tn_2d, fp_2d, fn_2d, tp_2d = cm_2d.ravel()
tn_full, fp_full, fn_full, tp_full = cm_full.ravel()

print(f"\n" + "="*40)
print("MÉTRICAS MÉDICAS CRÍTICAS")
print("="*40)

print(f"\nVersão 2D:")
print(f"- Verdadeiros Positivos (TP): {tp_2d} (câncer detectado corretamente)")
print(f"- Falsos Negativos (FN): {fn_2d} (câncer não detectado - PERIGOSO!)")
print(f"- Falsos Positivos (FP): {fp_2d} (alarme falso)")
print(f"- Verdadeiros Negativos (TN): {tn_2d} (benigno identificado corretamente)")

print(f"\nVersão 30D:")
print(f"- Verdadeiros Positivos (TP): {tp_full} (câncer detectado corretamente)")
print(f"- Falsos Negativos (FN): {fn_full} (câncer não detectado - PERIGOSO!)")
print(f"- Falsos Positivos (FP): {fp_full} (alarme falso)")
print(f"- Verdadeiros Negativos (TN): {tn_full} (benigno identificado corretamente)")

# Sensibilidade e Especificidade
sensitivity_2d = tp_2d / (tp_2d + fn_2d) if (tp_2d + fn_2d) > 0 else 0
specificity_2d = tn_2d / (tn_2d + fp_2d) if (tn_2d + fp_2d) > 0 else 0

sensitivity_full = tp_full / (tp_full + fn_full) if (tp_full + fn_full) > 0 else 0
specificity_full = tn_full / (tn_full + fp_full) if (tn_full + fp_full) > 0 else 0

print(f"\nMétricas Clínicas:")
print(f"2D - Sensibilidade (detectar câncer): {sensitivity_2d:.2%}")
print(f"2D - Especificidade (evitar falso alarme): {specificity_2d:.2%}")
print(f"30D - Sensibilidade (detectar câncer): {sensitivity_full:.2%}")
print(f"30D - Especificidade (evitar falso alarme): {specificity_full:.2%}")

# VISUALIZAÇÕES
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Subplot 1: Regiões de Decisão (apenas 2D)
X_combined_2d = np.vstack((X_train_2d_std, X_test_2d_std))
y_combined = np.hstack((y_train, y_test))

plt.subplot(2, 3, 1)
plot_decision_regions(X_combined_2d, y_combined, classifier=ppn_2d)
plt.title(f'Regiões de Decisão (2D)\nAcurácia: {test_accuracy_2d:.2%}')
plt.xlabel(f'{feature_names_2d[0]} (normalizada)')
plt.ylabel(f'{feature_names_2d[1]} (normalizada)')

# Subplot 2: Convergência 2D
plt.subplot(2, 3, 2)
plt.plot(range(1, len(ppn_2d.errors_history) + 1), ppn_2d.errors_history, 
         marker='o', linewidth=2, markersize=4, label='2D', color='blue')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência 2D')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3: Convergência 30D
plt.subplot(2, 3, 3)
plt.plot(range(1, len(ppn_full.errors_history) + 1), ppn_full.errors_history, 
         marker='s', linewidth=2, markersize=4, label='30D', color='green')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência 30D')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 4: Matriz de Confusão 2D
plt.subplot(2, 3, 4)
sns.heatmap(cm_2d, annot=True, fmt='d', cmap='Blues', 
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title('Matriz de Confusão (2D)')
plt.ylabel('Valor Real')
plt.xlabel('Predição')

# Subplot 5: Matriz de Confusão 30D
plt.subplot(2, 3, 5)
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens', 
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title('Matriz de Confusão (30D)')
plt.ylabel('Valor Real')
plt.xlabel('Predição')

# Subplot 6: Comparação de Acurácia
plt.subplot(2, 3, 6)
versions = ['2D\n(2 features)', '30D\n(30 features)']
accuracies = [test_accuracy_2d, test_accuracy_full]
colors = ['skyblue', 'lightgreen']
bars = plt.bar(versions, accuracies, color=colors, alpha=0.7)
plt.ylim(0, 1)
plt.ylabel('Acurácia')
plt.title('Comparação de Desempenho')
# Adicionar valores nos topos das barras
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ANÁLISE FINAL
print(f"\n" + "="*60)
print("ANÁLISE E CONCLUSÕES")
print("="*60)

print(f"\n1. Comparação 2D vs 30D:")
improvement = ((test_accuracy_full - test_accuracy_2d) / test_accuracy_2d) * 100
print(f"   - Acurácia 2D: {test_accuracy_2d:.2%}")
print(f"   - Acurácia 30D: {test_accuracy_full:.2%}")
print(f"   - Melhoria: {improvement:.1f}%")
print(f"   - Conclusão: {'30D é significativamente melhor' if improvement > 10 else '30D é um pouco melhor' if improvement > 2 else 'Desempenho similar'}")

print(f"\n2. Adequação para diagnóstico médico:")
if fn_2d == 0 and fn_full == 0:
    print(f"   ✓ EXCELENTE - Nenhum falso negativo (câncer não detectado)")
elif fn_2d <= 2 and fn_full <= 2:
    print(f"   ≈ ACEITÁVEL - Poucos falsos negativos ({fn_2d} em 2D, {fn_full} em 30D)")
else:
    print(f"   ✗ PROBLEMÁTICO - Muitos falsos negativos ({fn_2d} em 2D, {fn_full} em 30D)")

print(f"\n3. Características observadas:")
print(f"   - Dataset linearmente separável: {'PARCIALMENTE' if max(test_accuracy_2d, test_accuracy_full) > 0.85 else 'NÃO'}")
print(f"   - Convergência: 2D={'SIM' if 0 in ppn_2d.errors_history else 'NÃO'}, 30D={'SIM' if 0 in ppn_full.errors_history else 'NÃO'}")
print(f"   - Tempo de treinamento: 2D={training_time_2d:.4f}s, 30D={training_time_full:.4f}s")

print(f"\n4. Implicações médicas:")
print(f"   - Sensibilidade (detectar câncer): crítica para salvar vidas")
print(f"   - Especificidade (evitar falsos alarmes): importante para reduzir ansiedade")
print(f"   - Falsos Negativos são MUITO mais perigosos que Falsos Positivos")
print(f"   - Recomendação: usar 30D para melhor detecção")

print(f"\n5. Limitações do Perceptron para medicina:")
print(f"   - Algoritmo muito simples para diagnóstico complexo")
print(f"   - Não fornece probabilidades ou grau de confiança")
print(f"   - Melhor usar: Random Forest, SVM, ou Redes Neurais profundas")
print(f"   - Necessário validação clínica rigorosa antes do uso real")