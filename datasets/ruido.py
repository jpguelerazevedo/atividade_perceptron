# Exercício 4: Dataset de Classificação com Ruído
# Objetivo: Trabalhar com dados que têm sobreposição entre classes
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time

# Adicionar pasta pai ao path para importar perceptron e util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perceptron import Perceptron
from util import plot_decision_regions

# FUNÇÃO PARA TESTAR DIFERENTES CONFIGURAÇÕES
def test_configuration(class_sep, flip_y, title_suffix):
    """Testa uma configuração específica de separação e ruído"""
    
    print(f"\n" + "="*50)
    print(f"TESTE: {title_suffix}")
    print(f"Separação entre classes: {class_sep}")
    print(f"Ruído nos rótulos: {flip_y*100:.1f}%")
    print("="*50)
    
    # Gerar dataset
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,  # Controla separação
        flip_y=flip_y,        # Ruído nos rótulos
        random_state=42
    )
    
    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Treinar Perceptron
    start_time = time.time()
    ppn = Perceptron(learning_rate=0.01, n_epochs=100)
    ppn.fit(X_train_std, y_train)
    training_time = time.time() - start_time
    
    # Avaliar
    y_pred_test = ppn.predict(X_test_std)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Verificar convergência
    converged = 0 in ppn.errors_history
    conv_epoch = ppn.errors_history.index(0) + 1 if converged else None
    
    print(f"Resultados:")
    print(f"- Acurácia: {test_accuracy:.2%}")
    print(f"- Convergiu: {'SIM' if converged else 'NÃO'} {f'(época {conv_epoch})' if converged else ''}")
    print(f"- Erros finais: {ppn.errors_history[-1]}")
    print(f"- Tempo: {training_time:.4f}s")
    
    return X, y, X_train_std, X_test_std, y_train, y_test, ppn, test_accuracy, training_time

# PASSO 1: Configuração Inicial
print("=" * 60)
print("EXERCÍCIO 4: DATASET COM RUÍDO E SOBREPOSIÇÃO")
print("=" * 60)

# PASSO 2: Experimentos com diferentes configurações
results = []

# Experimento 1: Alta separação, sem ruído
X1, y1, X_train1, X_test1, y_train1, y_test1, ppn1, acc1, time1 = test_configuration(
    class_sep=3.0, flip_y=0.0, title_suffix="ALTA SEPARAÇÃO, SEM RUÍDO"
)
results.append(("Alta Sep, Sem Ruído", acc1, ppn1.errors_history[-1], time1, 0 in ppn1.errors_history))

# Experimento 2: Separação média, sem ruído  
X2, y2, X_train2, X_test2, y_train2, y_test2, ppn2, acc2, time2 = test_configuration(
    class_sep=1.5, flip_y=0.0, title_suffix="SEPARAÇÃO MÉDIA, SEM RUÍDO"
)
results.append(("Sep Média, Sem Ruído", acc2, ppn2.errors_history[-1], time2, 0 in ppn2.errors_history))

# Experimento 3: Baixa separação, sem ruído
X3, y3, X_train3, X_test3, y_train3, y_test3, ppn3, acc3, time3 = test_configuration(
    class_sep=0.5, flip_y=0.0, title_suffix="BAIXA SEPARAÇÃO, SEM RUÍDO"
)
results.append(("Baixa Sep, Sem Ruído", acc3, ppn3.errors_history[-1], time3, 0 in ppn3.errors_history))

# Experimento 4: Separação média, ruído baixo
X4, y4, X_train4, X_test4, y_train4, y_test4, ppn4, acc4, time4 = test_configuration(
    class_sep=1.5, flip_y=0.05, title_suffix="SEPARAÇÃO MÉDIA, RUÍDO 5%"
)
results.append(("Sep Média, Ruído 5%", acc4, ppn4.errors_history[-1], time4, 0 in ppn4.errors_history))

# Experimento 5: Separação média, ruído alto
X5, y5, X_train5, X_test5, y_train5, y_test5, ppn5, acc5, time5 = test_configuration(
    class_sep=1.5, flip_y=0.15, title_suffix="SEPARAÇÃO MÉDIA, RUÍDO 15%"
)
results.append(("Sep Média, Ruído 15%", acc5, ppn5.errors_history[-1], time5, 0 in ppn5.errors_history))

# PASSO 3: Resumo dos Resultados
print(f"\n" + "="*60)
print("RESUMO COMPARATIVO DOS EXPERIMENTOS")
print("="*60)

print(f"{'Experimento':<25} {'Acurácia':<10} {'Convergiu':<10} {'Erros':<8} {'Tempo(s)':<10}")
print("-" * 65)

for name, acc, errors, time_taken, converged in results:
    conv_str = "SIM" if converged else "NÃO"
    print(f"{name:<25} {acc:<10.2%} {conv_str:<10} {errors:<8} {time_taken:<10.4f}")

# PASSO 4: Visualizações Comparativas
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Datasets originais (linha 1)
datasets = [(X1, y1, "Alta Sep, Sem Ruído"), (X2, y2, "Sep Média, Sem Ruído"), (X3, y3, "Baixa Sep, Sem Ruído")]
for i, (X, y, title) in enumerate(datasets):
    plt.subplot(3, 3, i+1)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', alpha=0.7, label='Classe 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', alpha=0.7, label='Classe 1')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Regiões de decisão (linha 2)
perceptrons = [ppn1, ppn2, ppn3]
X_combined_list = [
    np.vstack((X_train1, X_test1)),
    np.vstack((X_train2, X_test2)), 
    np.vstack((X_train3, X_test3))
]
y_combined_list = [
    np.hstack((y_train1, y_test1)),
    np.hstack((y_train2, y_test2)),
    np.hstack((y_train3, y_test3))
]

for i, (ppn, X_comb, y_comb, (_, _, title)) in enumerate(zip(perceptrons, X_combined_list, y_combined_list, datasets)):
    plt.subplot(3, 3, i+4)
    plot_decision_regions(X_comb, y_comb, classifier=ppn)
    plt.title(f'Decisão: {title}')
    plt.xlabel('Feature 1 (norm)')
    plt.ylabel('Feature 2 (norm)')

# Convergência (linha 3)
all_perceptrons = [ppn1, ppn2, ppn3, ppn4, ppn5]
all_titles = ["Alta Sep", "Sep Média", "Baixa Sep", "Ruído 5%", "Ruído 15%"]

# Primeiro gráfico de convergência
plt.subplot(3, 3, 7)
for i, (ppn, title) in enumerate(zip(all_perceptrons[:3], all_titles[:3])):
    plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, 
             marker='o', label=title, linewidth=2, markersize=4)
plt.xlabel('Épocas')
plt.ylabel('Erros')
plt.title('Convergência: Efeito da Separação')
plt.legend()
plt.grid(True, alpha=0.3)

# Segundo gráfico de convergência (efeito do ruído)
plt.subplot(3, 3, 8)
for i, ppn in enumerate([ppn2, ppn4, ppn5]):
    labels = ["Sem Ruído", "Ruído 5%", "Ruído 15%"]
    plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, 
             marker='s', label=labels[i], linewidth=2, markersize=4)
plt.xlabel('Épocas')
plt.ylabel('Erros')
plt.title('Convergência: Efeito do Ruído')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico de barras com acurácias
plt.subplot(3, 3, 9)
names = [name.replace(", ", "\n") for name, _, _, _, _ in results]
accuracies = [acc for _, acc, _, _, _ in results]
colors = ['green', 'lightgreen', 'orange', 'lightcoral', 'red']
bars = plt.bar(names, accuracies, color=colors, alpha=0.7)
plt.ylabel('Acurácia')
plt.title('Comparação Final')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
# Adicionar valores nas barras
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# PASSO 5: Análise Detalhada dos Efeitos
print(f"\n" + "="*60)
print("ANÁLISE DETALHADA DOS EFEITOS")
print("="*60)

print(f"\n1. Efeito da Separação entre Classes:")
print(f"   - Alta separação (3.0): {acc1:.2%} - {'Convergiu' if 0 in ppn1.errors_history else 'Não convergiu'}")
print(f"   - Separação média (1.5): {acc2:.2%} - {'Convergiu' if 0 in ppn2.errors_history else 'Não convergiu'}")
print(f"   - Baixa separação (0.5): {acc3:.2%} - {'Convergiu' if 0 in ppn3.errors_history else 'Não convergiu'}")
print(f"   Conclusão: Quanto maior a separação, melhor o desempenho")

print(f"\n2. Efeito do Ruído nos Rótulos:")
print(f"   - Sem ruído (0%): {acc2:.2%} - {'Convergiu' if 0 in ppn2.errors_history else 'Não convergiu'}")
print(f"   - Ruído baixo (5%): {acc4:.2%} - {'Convergiu' if 0 in ppn4.errors_history else 'Não convergiu'}")  
print(f"   - Ruído alto (15%): {acc5:.2%} - {'Convergiu' if 0 in ppn5.errors_history else 'Não convergiu'}")
print(f"   Conclusão: Ruído nos rótulos prejudica significativamente o desempenho")

print(f"\n3. Convergência:")
convergence_count = sum(1 for _, _, _, _, converged in results if converged)
print(f"   - {convergence_count}/{len(results)} experimentos convergiram")
print(f"   - Condição necessária: dados devem ser linearmente separáveis")
print(f"   - Ruído impede convergência mesmo com boa separação")

print(f"\n4. Tempo de Treinamento:")
avg_time = np.mean([time_taken for _, _, _, time_taken, _ in results])
print(f"   - Tempo médio: {avg_time:.4f} segundos")
print(f"   - Muito rápido para todos os casos testados")
print(f"   - Perceptron não converge mais devagar com ruído, apenas não converge")

# PASSO 6: Implementação de Early Stopping (Bônus)
print(f"\n" + "="*60)
print("IMPLEMENTAÇÃO DE EARLY STOPPING")
print("="*60)

class PerceptronWithEarlyStopping(Perceptron):
    def __init__(self, learning_rate=0.01, n_epochs=100, patience=10):
        super().__init__(learning_rate, n_epochs)
        self.patience = patience
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Versão com early stopping baseada em validação"""
        if X_val is None or y_val is None:
            # Se não há conjunto de validação, usar o método original
            return super().fit(X, y)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_history = []
        self.val_accuracy_history = []
        
        best_val_accuracy = 0
        epochs_without_improvement = 0
        
        for epoch in range(self.n_epochs):
            errors = 0
            
            # Treinamento
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                error = y[idx] - y_predicted
                update = self.learning_rate * error
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0.0)
            
            self.errors_history.append(errors)
            
            # Validação
            val_predictions = self.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            self.val_accuracy_history.append(val_accuracy)
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= self.patience:
                print(f"Early stopping na época {epoch + 1} (sem melhoria por {self.patience} épocas)")
                print(f"Melhor acurácia de validação: {best_val_accuracy:.2%}")
                break

# Testar early stopping
print(f"\nTestando Early Stopping com dataset ruidoso:")
X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
    X_train4, y_train4, test_size=0.3, random_state=42, stratify=y_train4
)

ppn_es = PerceptronWithEarlyStopping(learning_rate=0.01, n_epochs=100, patience=5)
ppn_es.fit(X_train_es, y_train_es, X_val_es, y_val_es)

y_pred_es = ppn_es.predict(X_test4)
accuracy_es = accuracy_score(y_test4, y_pred_es)
print(f"Acurácia com early stopping: {accuracy_es:.2%}")

# PASSO 7: Conclusões Finais
print(f"\n" + "="*60)
print("CONCLUSÕES FINAIS")
print("="*60)

print(f"\n1. Fatores críticos para o sucesso do Perceptron:")
print(f"   ✓ Separação entre classes (>= 1.5 recomendado)")
print(f"   ✓ Ausência de ruído nos rótulos")
print(f"   ✓ Dados linearmente separáveis")

print(f"\n2. Quando o Perceptron falha:")
print(f"   ✗ Classes muito próximas (sobreposição)")
print(f"   ✗ Ruído nos rótulos > 10%")
print(f"   ✗ Dados não-linearmente separáveis")

print(f"\n3. Estratégias para melhorar robustez:")
print(f"   - Early stopping com conjunto de validação")
print(f"   - Pré-processamento mais cuidadoso dos dados")
print(f"   - Usar algoritmos mais robustos (SVM, Random Forest)")
print(f"   - Múltiplas execuções com diferentes seeds")

print(f"\n4. Aplicabilidade prática:")
print(f"   - Bom para problemas simples e bem separados")
print(f"   - Inadequado para dados ruidosos do mundo real")
print(f"   - Útil como baseline ou para entender conceitos")
print(f"   - Base para algoritmos mais sofisticados")

print(f"\n5. Lição principal:")
print(f"   A qualidade dos dados é FUNDAMENTAL para o sucesso.")
print(f"   'Garbage in, garbage out' - dados ruins levam a modelos ruins.")