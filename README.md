# Como executar o projeto 

### Crie um ambiente virtual

Criar ambientes virtuais é util pra evitar problemas de versao entre bibliotecas usadas pra diferentes projetos além de nao instalar um monte de coisa desnecessária deixando seu PC uma carroça.
* `python -m venv nome_do_ambiente`
* `nome_do_ambiente\Scripts\activate`

Para insalar a bibliotecas use:
* `pip install nome-da-biblioteca`

Pronto já pode rodar o projeto. 

# O que esse código faz ? 

Vou deixar aqui o link pra uma doc que vi no [Medium](https://medium.com/@abhishekjainindore24/understanding-the-1d-convolutional-layer-in-deep-learning-7a4cb994c981) bem legal e um [tutorial](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/).

A [base dados](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset/data) que eu estou usando foi baixada da plataforma Kaggle.



# Breast Cancer Diagnosis with 1D CNN

Este projeto utiliza um algoritmo de rede neural convolucional 1D (CNN) para classificar tumores de mama como malignos ou benignos, usando o conjunto de dados Breast Cancer Wisconsin (Diagnostic).

## Requisitos

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

Instale as dependências com:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## Passo a Passo

### 1. Carregamento e Pré-processamento dos Dados
- O arquivo `Breast_cancer_dataset.csv` é carregado.
- Colunas irrelevantes (`id`, `Unnamed: 32`) são removidas.
- A coluna `diagnosis` é convertida para valores binários: `M` (maligno) = 1, `B` (benigno) = 0.
- Os dados são divididos em features (`X`) e rótulos (`y`).

### 2. Divisão dos Dados
- Os dados são divididos em treino, validação e teste (80% treino/validação, 20% teste; depois 75% treino, 25% validação).
- As features são normalizadas com MinMaxScaler.
- As matrizes de features são convertidas para o formato 3D esperado pela camada Conv1D.

### 3. Construção do Modelo
- O modelo é uma rede sequencial com as seguintes camadas:
  - Conv1D (32 filtros, kernel 5, ReLU)
  - MaxPooling1D
  - Dropout
  - Conv1D (64 filtros, kernel 5, ReLU)
  - GlobalMaxPooling1D
  - Dense (64 unidades, ReLU)
  - Dropout
  - Dense (1 unidade, Sigmoid)
- Métricas monitoradas: AUC ROC, AUC PR, Acurácia binária.

### 4. Treinamento
- O modelo é treinado por até 200 épocas, com early stopping e redução de taxa de aprendizado.
- O melhor modelo (maior AUC de validação) é salvo em `checkpoints/best_val_auc.keras`.

### 5. Avaliação
- Probabilidades de predição são geradas para validação e teste.
- O melhor limiar de decisão é escolhido para maximizar o F1 na validação.
- Métricas detalhadas são calculadas: ROC AUC, PR AUC, precisão, revocação, F1, matriz de confusão, etc.
- Relatórios de classificação são impressos para ambos limiares (0.5 e F1-max).

### 6. Visualização
- Curvas ROC e PR são geradas e salvas em `wdbc_roc_pr_test.pdf`.
- Gráficos de acurácia/perda do treinamento são salvos em `acuracia_perda.pdf`.
- Matriz de confusão do teste é salva em `matriz_confusao.pdf`.
- Mapa de calor de correlação entre features é salvo em `mapa_calor.pdf`.

## Execução

Execute o script principal:
```bash
python index.py
```

Os arquivos de saída (PDFs e PNGs) serão gerados na pasta do projeto.

## Estrutura dos Arquivos
- `index.py`: Script principal.
- `Breast_cancer_dataset.csv`: Base de dados.
- `checkpoints/`: Modelos salvos.
- Arquivos PDF/PNG: Resultados das análises e gráficos.

## Referências
- [Breast Cancer Wisconsin (Diagnostic) Data Set - UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Documentação das bibliotecas utilizadas.