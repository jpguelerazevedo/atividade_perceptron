# Relatório Final - Atividade Prática Perceptron

**Dupla:** Leonardo Rodrigues Cavalcante & João Paulo Gueller Azevedo  

---

## Exercício 1: Iris Dataset (Setosa vs Versicolor)

### 1. Descrição do Dataset
- **Número de amostras:** 100 (50 de cada classe)
- **Número de features:** 2 (sepal length, petal length)
- **Distribuição das classes:** Balanceada (50 setosa, 50 versicolor)
- **É linearmente separável:** SIM

### 2. Resultados
- **Acurácia no treino:** 100.00%
- **Acurácia no teste:** 100.00%
- **Número de épocas até convergência:** 2
- **Tempo de treinamento:** 0.0010 segundos

### 3. Visualizações
- **Gráfico de convergência:** Erros chegaram a zero na época 2
- **Regiões de decisão:** Separação linear perfeita entre as classes
- **Matriz de confusão:** [[15, 0], [0, 15]] - Sem erros de classificação

### 4. Análise
- **O perceptron foi adequado para este problema:** SIM. O algoritmo obteve 100% de acurácia e convergiu rapidamente.
- **Melhorias sugeridas:** Usar todas as 4 features do dataset para melhor representação, testar com as 3 classes do Iris.
- **Comparação com expectativas:** Resultado conforme esperado, pois Setosa e Versicolor são linearmente separáveis.

---

## Exercício 2: Moons Dataset

### 1. Descrição do Dataset
- **Número de amostras:** 200 (100 de cada classe)
- **Número de features:** 2
- **Distribuição das classes:** Balanceada (100 lua inferior, 100 lua superior)
- **É linearmente separável:** NÃO (formato de luas entrelaçadas)

### 2. Resultados
- **Acurácia no treino:** 80.71%
- **Acurácia no teste:** 90.00%
- **Número de épocas até convergência:** Não convergiu (100 épocas)
- **Tempo de treinamento:** 0.1080 segundos

### 3. Visualizações
- **Gráfico de convergência:** Erros oscilaram mas nunca chegaram a zero
- **Regiões de decisão:** Linha reta inadequada para separar formato curvo
- **Matriz de confusão:** [[27, 3], [3, 27]] - 6 erros de classificação

### 4. Análise
- **O perceptron foi adequado para este problema:** NÃO. Dados não-linearmente separáveis requerem fronteira curva.
- **Melhorias sugeridas:** Usar Perceptron Multicamadas (MLP), SVM com kernel RBF, ou Redes Neurais.
- **Comparação com expectativas:** Acurácia superior ao esperado (90% vs 50-60%), mas ainda inadequado.

---

## Exercício 3: Breast Cancer Wisconsin

### 1. Descrição do Dataset
- **Número de amostras:** 569 (212 malignant, 357 benign)
- **Número de features:** 30 (versão completa) / 2 (versão visualização)
- **Distribuição das classes:** Desbalanceada (37% malignant, 63% benign)
- **É linearmente separável:** Parcialmente (depende das features usadas)

### 2. Resultados
**Versão 2D:**
- **Acurácia no treino:** 88.94%
- **Acurácia no teste:** 85.96%
- **Número de épocas até convergência:** Não convergiu
- **Tempo de treinamento:** 0.2941 segundos

**Versão 30D:**
- **Acurácia no treino:** 98.99%
- **Acurácia no teste:** 96.49%
- **Número de épocas até convergência:** Não convergiu
- **Tempo de treinamento:** 0.2936 segundos

### 3. Visualizações
- **Gráfico de convergência:** Ambas versões não convergiram, mas 30D teve menos erros
- **Regiões de decisão:** Disponível apenas para versão 2D
- **Matriz de confusão:** 2D: [[60, 4], [20, 87]] vs 30D: [[59, 5], [1, 106]]

### 4. Análise
- **O perceptron foi adequado para este problema:** PARCIALMENTE. Versão 30D muito melhor que 2D.
- **Melhorias sugeridas:** Usar algoritmos mais robustos (Random Forest, SVM), implementar validação clínica rigorosa.
- **Comparação com expectativas:** 30D superou expectativas (96.49%), 2D foi limitada (85.96%).

---

## Exercício 4: Dataset de Classificação com Ruído

### 1. Descrição do Dataset
- **Número de amostras:** 200 por experimento
- **Número de features:** 2
- **Distribuição das classes:** Balanceada
- **É linearmente separável:** Varia conforme separação e ruído

### 2. Resultados
| Experimento | Acurácia | Convergiu | Épocas | Tempo (s) |
|-------------|----------|-----------|---------|-----------|
| Alta Sep, Sem Ruído | 100.00% | NÃO | - | 0.1224 |
| Sep Média, Sem Ruído | 95.00% | NÃO | - | 0.1256 |
| Baixa Sep, Sem Ruído | 45.00% | NÃO | - | 0.1128 |
| Sep Média, Ruído 5% | 90.00% | NÃO | - | 0.1120 |
| Sep Média, Ruído 15% | 88.33% | NÃO | - | 0.1149 |

### 3. Visualizações
- **Gráfico de convergência:** Nenhum experimento convergiu completamente
- **Regiões de decisão:** Qualidade da separação varia com parâmetros
- **Matriz de confusão:** Disponível para cada configuração

### 4. Análise
- **O perceptron foi adequado para este problema:** Depende da configuração. Melhor com alta separação e sem ruído.
- **Melhorias sugeridas:** Early stopping, pré-processamento cuidadoso, algoritmos mais robustos.
- **Comparação com expectativas:** Confirmou que separação e ausência de ruído são críticas.

---

## Exercício 5: Dataset Linearmente Separável Personalizado

### 1. Descrição do Dataset
- **Número de amostras:** 100 por experimento (50 de cada classe)
- **Número de features:** 2
- **Distribuição das classes:** Balanceada
- **É linearmente separável:** SIM (controlado pela distância entre centros)

### 2. Resultados
| Experimento | Distância | Acurácia | Convergiu | Épocas | Tempo (s) |
|-------------|-----------|----------|-----------|---------|-----------|
| Separação Excelente | 8.49 | 100.00% | SIM | 2 | <0.1 |
| Separação Boa | 4.47 | 100.00% | SIM | 2 | <0.1 |
| Separação Limítrofe | 3.16 | 96.67% | NÃO | - | <0.1 |

### 3. Visualizações
- **Gráfico de convergência:** Experimentos 1 e 2 convergiram rapidamente
- **Regiões de decisão:** Fronteiras lineares claras para todos os casos
- **Matriz de confusão:** Erros mínimos nos experimentos bem separados

### 4. Análise
- **O perceptron foi adequado para este problema:** SIM para experimentos com boa separação.
- **Melhorias sugeridas:** Usar distância mínima de 2.0 unidades entre centros para garantir convergência.

- **Comparação com expectativas:** Confirmou relação direta entre separação geométrica e desempenho.


