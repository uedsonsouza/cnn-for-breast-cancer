# Como executar o projeto 

### Crie um ambiente virtual

Criar ambientes virtuais é util pra evitar problemas de versao entre bibliotecas usadas pra diferentes projetos além de nao instalar um monte de coisa desnecessária deixando seu PC uma carroça.
* `python -m venv nome_do_ambiente`
* `nome_do_ambiente\Scripts\activate`

Para insalar a bibliotecas use:
* `pip install nome-da-biblioteca`

Pronto já pode rodar o projeto. 

# O que esse código faz ? 

Vou deixar aqui o link pra uma doc que vi no [Medium](https://medium.com/@abhishekjainindore24/understanding-the-1d-convolutional-layer-in-deep-learning-7a4cb994c981) bem legal.
A [base dados](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset/data) que eu estou usando foi baixada da plataforma Kaggle.

- `MinMaxScaler`: Para normalização de dados (escala 0-1)
- `LabelEncoder`: Para codificação de labels categóricos em numéricos
- `classification_report`: Relatório com precisão, recall, F1-score
- `confusion_matrix`: Matriz de confusão para análise de classificação

## Arquitetura Final da CNN-1D

```
Entrada: (batch_size, n_features, 1)
    ↓
Conv1D (32 filtros) + ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (20%)
    ↓
Conv1D (64 filtros) + ReLU
    ↓
GlobalMaxPooling1D
    ↓
Dense (64 neurônios) + ReLU
    ↓
Dropout (10%)
    ↓
Dense (1 neurônio) + Sigmoid
    ↓
Saída: Probabilidade de malignidade [0,1]
```