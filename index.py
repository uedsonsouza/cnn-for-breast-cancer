import pandas as pd
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Definir modelo CNN-1D
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Treinar modelo
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 4. Avaliação
preds = model.predict(X_test)
y_pred = (preds > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Grafico pra visualizacao dos resultados

plt.title('Gráfico de Acurácia e Perda')
plt.xlabel('Épocas')
plt.ylabel('Acurácia / Perda')
plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.legend()
plt.show()

df_clean = df.copy()
sns.set_theme(style="darkgrid", palette="coolwarm")
# Esse gráfico nao entra no artigo devido visualização ruim
sns.pairplot(df_clean, vars=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean'], hue='diagnosis', markers=["o", "s"], plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot de Features Selecionadas por Diagnóstico', y=1.02)
plt.show()

plt.figure(figsize=(12,10))
corr = df_clean.drop('diagnosis', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Mapa de Calor de Correlação entre Features')
plt.tight_layout()
plt.show()