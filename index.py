import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar e preparar os dados
file_path = 'Breast_cancer_dataset.csv'
df = pd.read_csv(file_path)

# Remover colunas desnecessárias
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# usando o LabelEncoder para transformar M=1, B=0
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Separar features e alvo
X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

# Normalizar os dados
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Redimensionar para entrada da CNN-1D (amostras, features, canais)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Separar em treino/teste
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val)
# 2. Definir modelo CNN-1D
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Treinar modelo
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# 4. Avaliação
preds = model.predict(X_test)
y_pred = (preds > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Aqui vou printar a matriz de confusao
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Grafico pra visualizacao dos resultados
plt.title('Gráfico de Acurácia e Perda')
plt.xlabel('Épocas')
plt.ylabel('Acurácia / Perda')
plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.plot(history.history['loss'], label='Perda Treinamento')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.legend()
print('TENHO CUM CADIM DE DADOS --->',history.history.keys())
plt.show()

plt.figure(figsize=(10, 6))
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# gero o plot pra matriz de confusao com percentuais 
plt.title('Matriz de confusão')
annotations = []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annotations.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')

annotations = np.array(annotations).reshape(cm.shape)
sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
            xticklabels=['Benigno ', 'Maligno'],
            yticklabels=['Benigno ', 'Maligno'])

df_clean = df.copy()

plt.figure(figsize=(12,10))
corr = df_clean.drop('diagnosis', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Mapa de Calor de Correlação entre Features')
plt.tight_layout()
plt.show()