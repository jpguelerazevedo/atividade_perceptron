# Exercício 5: Dataset Linearmente Separável Personalizado (DLSP)
# Objetivo: Criar seu próprio dataset e entender a geometria da solução
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time

# Adicionar pasta pai ao path para importar perceptron e util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perceptron import Perceptron
from util import plot_decision_regions

# PASSO 1: Criar Dataset Customizado
print("=" * 60)
print("EXERCÍCIO 5: DATASET LINEARMENTE SEPARÁVEL PERSONALIZADO")
print("=" * 60)

def create_custom_dataset(center_0, center_1, std=1.0, n_samples_per_class=50, seed=42):
    """Cria um dataset customizado com dois centros especificados"""
    np.random.seed(seed)
    
    # Classe 0: centro especificado
    class_0 = np.random.normal(center_0, std, (n_samples_per_class, 2))
    
    # Classe 1: centro especificado  
    class_1 = np.random.normal(center_1, std, (n_samples_per_class, 2))
    
    # Combinar
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    return X, y, class_0, class_1

def analyze_separation(center_0, center_1):
    """Analisa a separação teórica entre os centros"""
    distance = np.linalg.norm(np.array(center_1) - np.array(center_0))
    return distance

# EXPERIMENTO 1: Separação Muito Boa
print(f"\n" + "="*50)
print("EXPERIMENTO 1: SEPARAÇÃO MUITO BOA")
print("="*50)

center_0_exp1 = [-3, -3]
center_1_exp1 = [3, 3]
distance_exp1 = analyze_separation(center_0_exp1, center_1_exp1)

X1, y1, class_0_exp1, class_1_exp1 = create_custom_dataset(center_0_exp1, center_1_exp1)

print(f"Configuração:")
print(f"- Centro Classe 0: {center_0_exp1}")
print(f"- Centro Classe 1: {center_1_exp1}")
print(f"- Distância entre centros: {distance_exp1:.2f}")
print(f"- Desvio padrão: 1.0")
print(f"- Separabilidade teórica: EXCELENTE")

# EXPERIMENTO 2: Separação Média
print(f"\n" + "="*50)
print("EXPERIMENTO 2: SEPARAÇÃO MÉDIA")
print("="*50)

center_0_exp2 = [-2, -1]
center_1_exp2 = [2, 1]
distance_exp2 = analyze_separation(center_0_exp2, center_1_exp2)

X2, y2, class_0_exp2, class_1_exp2 = create_custom_dataset(center_0_exp2, center_1_exp2)

print(f"Configuração:")
print(f"- Centro Classe 0: {center_0_exp2}")
print(f"- Centro Classe 1: {center_1_exp2}")
print(f"- Distância entre centros: {distance_exp2:.2f}")
print(f"- Separabilidade teórica: BOA")

# EXPERIMENTO 3: Separação Limítrofe
print(f"\n" + "="*50)
print("EXPERIMENTO 3: SEPARAÇÃO LIMÍTROFE")
print("="*50)

center_0_exp3 = [-1.5, -0.5]
center_1_exp3 = [1.5, 0.5]
distance_exp3 = analyze_separation(center_0_exp3, center_1_exp3)

X3, y3, class_0_exp3, class_1_exp3 = create_custom_dataset(center_0_exp3, center_1_exp3)

print(f"Configuração:")
print(f"- Centro Classe 0: {center_0_exp3}")
print(f"- Centro Classe 1: {center_1_exp3}")
print(f"- Distância entre centros: {distance_exp3:.2f}")
print(f"- Separabilidade teórica: LIMÍTROFE")

# FUNÇÃO PARA TREINAR E AVALIAR
def train_and_evaluate(X, y, experiment_name):
    """Treina e avalia um experimento"""
    print(f"\nTreinando {experiment_name}...")
    
    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Treinar
    start_time = time.time()
    ppn = Perceptron(learning_rate=0.01, n_epochs=50)
    ppn.fit(X_train_std, y_train)
    training_time = time.time() - start_time
    
    # Avaliar
    y_pred_test = ppn.predict(X_test_std)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Convergência
    converged = 0 in ppn.errors_history
    conv_epoch = ppn.errors_history.index(0) + 1 if converged else None
    
    return {
        'X_train_std': X_train_std,
        'X_test_std': X_test_std,
        'y_train': y_train,
        'y_test': y_test,
        'ppn': ppn,
        'accuracy': test_accuracy,
        'training_time': training_time,
        'converged': converged,
        'conv_epoch': conv_epoch,
        'scaler': scaler
    }

# EXECUTAR TODOS OS EXPERIMENTOS
results_exp1 = train_and_evaluate(X1, y1, "Experimento 1")
results_exp2 = train_and_evaluate(X2, y2, "Experimento 2")
results_exp3 = train_and_evaluate(X3, y3, "Experimento 3")

# ANÁLISE GEOMÉTRICA DETALHADA
print(f"\n" + "="*60)
print("ANÁLISE GEOMÉTRICA DETALHADA")
print("="*60)

def geometric_analysis(results, X, experiment_name, center_0, center_1):
    """Análise geométrica completa"""
    ppn = results['ppn']
    
    print(f"\n{experiment_name}:")
    print(f"- Acurácia: {results['accuracy']:.2%}")
    convergence_text = f"(época {results['conv_epoch']})" if results['converged'] and results['conv_epoch'] else ""
    print(f"- Convergiu: {'SIM' if results['converged'] else 'NÃO'} {convergence_text}")
    
    # Pesos aprendidos
    print(f"\nPesos aprendidos:")
    print(f"- w1: {ppn.weights[0]:.4f}")
    print(f"- w2: {ppn.weights[1]:.4f}")
    print(f"- bias: {ppn.bias:.4f}")
    
    # Equação da fronteira
    if ppn.weights[1] != 0:
        slope = -ppn.weights[0]/ppn.weights[1]
        intercept = -ppn.bias/ppn.weights[1]
        print(f"\nEquação da fronteira de decisão:")
        print(f"x2 = {slope:.3f} * x1 + {intercept:.3f}")
    
    # Vetor normal à fronteira
    normal_vector = ppn.weights / np.linalg.norm(ppn.weights)
    print(f"\nVetor normal à fronteira (normalizado): [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}]")
    
    # Distância dos centros à fronteira
    center_0_norm = results['scaler'].transform([center_0])[0]
    center_1_norm = results['scaler'].transform([center_1])[0]
    
    dist_center_0 = abs(np.dot(center_0_norm, ppn.weights) + ppn.bias) / np.linalg.norm(ppn.weights)
    dist_center_1 = abs(np.dot(center_1_norm, ppn.weights) + ppn.bias) / np.linalg.norm(ppn.weights)
    
    print(f"\nDistâncias dos centros à fronteira:")
    print(f"- Centro Classe 0: {dist_center_0:.3f}")
    print(f"- Centro Classe 1: {dist_center_1:.3f}")
    print(f"- Margem total: {dist_center_0 + dist_center_1:.3f}")
    
    # Verificar classificação dos centros
    pred_center_0 = ppn.predict(center_0_norm.reshape(1, -1))[0]
    pred_center_1 = ppn.predict(center_1_norm.reshape(1, -1))[0]
    
    print(f"\nClassificação dos centros:")
    print(f"- Centro Classe 0 classificado como: {pred_center_0} ({'✓' if pred_center_0 == 0 else '✗'})")
    print(f"- Centro Classe 1 classificado como: {pred_center_1} ({'✓' if pred_center_1 == 1 else '✗'})")

# Executar análise geométrica
geometric_analysis(results_exp1, X1, "EXPERIMENTO 1", center_0_exp1, center_1_exp1)
geometric_analysis(results_exp2, X2, "EXPERIMENTO 2", center_0_exp2, center_1_exp2)
geometric_analysis(results_exp3, X3, "EXPERIMENTO 3", center_0_exp3, center_1_exp3)

# VISUALIZAÇÕES COMPREHENSIVAS
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

experiments = [
    (X1, y1, results_exp1, "Experimento 1: Separação Excelente", center_0_exp1, center_1_exp1),
    (X2, y2, results_exp2, "Experimento 2: Separação Boa", center_0_exp2, center_1_exp2),
    (X3, y3, results_exp3, "Experimento 3: Separação Limítrofe", center_0_exp3, center_1_exp3)
]

for row, (X, y, results, title, center_0, center_1) in enumerate(experiments):
    ppn = results['ppn']
    X_combined = np.vstack((results['X_train_std'], results['X_test_std']))
    y_combined = np.hstack((results['y_train'], results['y_test']))
    
    # Coluna 1: Dataset original
    plt.subplot(3, 4, row*4 + 1)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', alpha=0.7, label='Classe 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', alpha=0.7, label='Classe 1')
    plt.scatter(*center_0, c='darkred', marker='x', s=200, label=f'Centro 0')
    plt.scatter(*center_1, c='darkblue', marker='x', s=200, label=f'Centro 1')
    plt.title(f'{title}\nDataset Original')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Coluna 2: Regiões de decisão
    plt.subplot(3, 4, row*4 + 2)
    plot_decision_regions(X_combined, y_combined, classifier=ppn)
    plt.title(f'Regiões de Decisão\nAcurácia: {results["accuracy"]:.2%}')
    plt.xlabel('Feature 1 (normalizada)')
    plt.ylabel('Feature 2 (normalizada)')
    
    # Coluna 3: Convergência
    plt.subplot(3, 4, row*4 + 3)
    plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, 
             marker='o', linewidth=2, markersize=4)
    plt.xlabel('Épocas')
    plt.ylabel('Erros')
    plt.title(f'Convergência\n{"Convergiu" if results["converged"] else "Não convergiu"}')
    plt.grid(True, alpha=0.3)
    
    # Coluna 4: Geometria da fronteira
    plt.subplot(3, 4, row*4 + 4)
    plot_decision_regions(X_combined, y_combined, classifier=ppn)
    
    # Desenhar vetor normal
    if ppn.weights[1] != 0:
        # Ponto central da fronteira
        x_center = np.mean(X_combined[:, 0])
        y_center = (-ppn.weights[0] * x_center - ppn.bias) / ppn.weights[1]
        
        # Vetor normal normalizado
        normal = ppn.weights / np.linalg.norm(ppn.weights)
        scale = 1.0
        
        plt.arrow(x_center, y_center, normal[0]*scale, normal[1]*scale,
                 head_width=0.1, head_length=0.1, fc='yellow', ec='black', linewidth=3)
        plt.text(x_center + normal[0]*scale*1.2, y_center + normal[1]*scale*1.2,
                'Vetor\nNormal', ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.title(f'Geometria da Fronteira\nVetor Normal')
    plt.xlabel('Feature 1 (normalizada)')
    plt.ylabel('Feature 2 (normalizada)')

plt.tight_layout()
plt.show()

# TESTE DE LIMITES: Movendo os centros gradualmente
print(f"\n" + "="*60)
print("TESTE DE LIMITES: APROXIMANDO OS CENTROS")
print("="*60)

def test_convergence_limit():
    """Testa o limite de convergência aproximando os centros"""
    distances = np.arange(4.0, 1.0, -0.5)  # De 4.0 até 1.5
    results_limit = []
    
    for dist in distances:
        # Centros simétricos com distância específica
        center_0 = [-dist/2, -dist/2]
        center_1 = [dist/2, dist/2]
        
        X, y, _, _ = create_custom_dataset(center_0, center_1, std=0.8, n_samples_per_class=30)
        
        # Treinar rapidamente
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        
        ppn = Perceptron(learning_rate=0.01, n_epochs=30)
        ppn.fit(X_train_std, y_train)
        
        accuracy = accuracy_score(y_test, ppn.predict(X_test_std))
        converged = 0 in ppn.errors_history
        
        results_limit.append((dist, accuracy, converged))
        print(f"Distância {dist:.1f}: Acurácia={accuracy:.2%}, Convergiu={'SIM' if converged else 'NÃO'}")
    
    return results_limit

limit_results = test_convergence_limit()

# Visualizar teste de limites
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
distances = [r[0] for r in limit_results]
accuracies = [r[1] for r in limit_results]
converged = [r[2] for r in limit_results]

colors = ['green' if conv else 'red' for conv in converged]
plt.scatter(distances, accuracies, c=colors, s=100, alpha=0.7)
plt.plot(distances, accuracies, 'k--', alpha=0.5)
plt.xlabel('Distância entre Centros')
plt.ylabel('Acurácia')
plt.title('Acurácia vs Distância\n(Verde=Convergiu, Vermelho=Não)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
conv_count = [sum(converged[:i+1]) for i in range(len(converged))]
plt.bar(range(len(distances)), [1 if c else 0 for c in converged], 
        color=colors, alpha=0.7)
plt.xlabel('Experimento (Distância Decrescente)')
plt.ylabel('Convergiu (1=Sim, 0=Não)')
plt.title('Padrão de Convergência')
plt.xticks(range(len(distances)), [f'{d:.1f}' for d in distances])
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# CONCLUSÕES FINAIS
print(f"\n" + "="*60)
print("CONCLUSÕES FINAIS")
print("="*60)

print(f"\n1. Relação Distância-Desempenho:")
best_exp = max([(results_exp1['accuracy'], 1), (results_exp2['accuracy'], 2), (results_exp3['accuracy'], 3)])
print(f"   - Melhor experimento: {best_exp[1]} (Acurácia: {best_exp[0]:.2%})")
print(f"   - Padrão observado: Maior separação → Melhor desempenho")

print(f"\n2. Geometria da Solução:")
print(f"   - O Perceptron encontra um hiperplano (linha) que separa as classes")
print(f"   - O vetor de pesos é perpendicular à fronteira de decisão")
print(f"   - A solução não é única (várias linhas podem separar perfeitamente)")

print(f"\n3. Condições para Convergência:")
convergence_results = [results_exp1['converged'], results_exp2['converged'], results_exp3['converged']]
conv_count = sum(convergence_results)
print(f"   - {conv_count}/3 experimentos convergiram")
print(f"   - Convergência garantida apenas se dados são linearmente separáveis")
print(f"   - Separação mínima necessária: aproximadamente 2.0 unidades (com std=1.0)")

print(f"\n4. Insights Geométricos:")
print(f"   - Fronteira sempre passa 'entre' as nuvens de pontos")
print(f"   - Margem depende da separação e dispersão dos dados")
print(f"   - Centros bem separados facilitam a classificação")

print(f"\n5. Aplicações Práticas:")
print(f"   - Use este conhecimento para avaliar viabilidade antes do treinamento")
print(f"   - Visualize seus dados 2D antes de aplicar o Perceptron")
print(f"   - Se não há separação clara, considere outros algoritmos")

print(f"\n6. Extensões Possíveis:")
print(f"   - Testar com diferentes distribuições (não-gaussianas)")
print(f"   - Implementar Perceptron com margem (precursor do SVM)")
print(f"   - Experimentar com transformações não-lineares dos features")
print(f"   - Usar Perceptron Multicamadas para problemas mais complexos")

print(f"\n" + "="*60)
print("EXERCÍCIO 5 CONCLUÍDO COM SUCESSO!")
print("="*60)