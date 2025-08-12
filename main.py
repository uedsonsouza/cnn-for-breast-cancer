import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve, precision_recall_curve)

sns.set_palette("colorblind")
colors_blind_palette = sns.color_palette()
figsize_double_column_split = (7.0, 2.5)

def to_cnn1d(arr):
    """
    Convert a 2D feature matrix (N, F) into a 3D tensor (N, F, 1) expected by Conv1D.
    - N: number of samples
    - F: number of features (sequence length for Conv1D)
    - 1: single input channel
    """
    return np.expand_dims(arr, axis=-1)

def evaluate_at_threshold(y_true, probs, threshold=0.5):
    """
    Return a comprehensive set of metrics at a fixed threshold.
    Includes threshold-independent metrics (ROC AUC, PR AUC).
    Assumes positive class is 1 (malignant).
    """
    
    def safe_div(a, b):
        return float(a) / b if b else 0.0
    
    y_pred = (probs >= threshold).astype(int)
    
    roc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    tpr = recall
    tnr = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    ppv = precision
    npv = safe_div(tn, tn + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    
    prevalence = safe_div(tp + fn, tp + tn + fp + fn)
    pred_pos_rate = safe_div(tp + fp, tp + tn + fp + fn)
    pred_neg_rate = 1.0 - pred_pos_rate
    
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
    cm_norm = np.nan_to_num(cm_norm)
    
    return {
        'threshold': float(threshold),
        'roc_auc': float(roc),
        'pr_auc': float(auprc),
        'precision': float(ppv),
        'recall': float(tpr),
        'specificity': float(tnr),
        'npv': float(npv),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'prevalence': float(prevalence),
        'predicted_positive_rate': float(pred_pos_rate),
        'predicted_negative_rate': float(pred_neg_rate),
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_norm,
        'tn_fp_fn_tp': (int(tn), int(fp), int(fn), int(tp)),
    }

SEED = 42
file_path = 'data/Breast_cancer_dataset.csv'
df = pd.read_csv(file_path)
df = df.drop(columns=[c for c in ['id', 'Unnamed: 32'] if c in df.columns])

if df['diagnosis'].dtype == object:
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype('int8')

X = df.drop(columns=['diagnosis']).astype('float32').values
y = df['diagnosis'].astype('int8').values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train).astype('float32')
X_val = scaler.transform(X_val).astype('float32')
X_test = scaler.transform(X_test).astype('float32')

X_train = to_cnn1d(X_train)
X_val = to_cnn1d(X_val)
X_test = to_cnn1d(X_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

metrics = [
    tf.keras.metrics.AUC(curve='ROC', name='auc'),
    tf.keras.metrics.AUC(curve='PR', name='auprc'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=metrics
)

ckpt_dir = Path('checkpoints')
ckpt_dir.mkdir(exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_dir / 'best_val_auc.keras'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

val_probs = model.predict(X_val, verbose=0).ravel()
test_probs = model.predict(X_test, verbose=0).ravel()

thr_grid = np.linspace(0.01, 0.99, 99)
best_f1, best_thr_f1 = -1.0, 0.5
for t in thr_grid:
    y_pred_t = (val_probs >= t).astype(int)
    _, _, f1_t, _ = precision_recall_fscore_support(y_val, y_pred_t, average='binary', zero_division=0)
    if f1_t > best_f1:
        best_f1, best_thr_f1 = f1_t, t

val_default = evaluate_at_threshold(y_val, val_probs, threshold=0.5)
val_f1 = evaluate_at_threshold(y_val, val_probs, threshold=best_thr_f1)

print("\n=== Validation (threshold=0.5) ===")
print(f"ROC AUC: {val_default['roc_auc']:.4f} | PR AUC: {val_default['pr_auc']:.4f} | "
      f"P: {val_default['precision']:.4f} | R: {val_default['recall']:.4f} | F1: {val_default['f1']:.4f}")
print("Confusion matrix (val):\n", val_default['confusion_matrix'])
print("Confusion matrix normalized (val):\n", np.round(val_default['confusion_matrix_normalized'], 3))

print("\n=== Validation (F1-max threshold) ===")
print(f"Chosen thr (F1-max): {best_thr_f1:.3f}")
print(f"ROC AUC: {val_f1['roc_auc']:.4f} | PR AUC: {val_f1['pr_auc']:.4f} | "
      f"P: {val_f1['precision']:.4f} | R: {val_f1['recall']:.4f} | F1: {val_f1['f1']:.4f}")
print("Confusion matrix (val):\n", val_f1['confusion_matrix'])
print("Confusion matrix normalized (val):\n", np.round(val_f1['confusion_matrix_normalized'], 3))

test_default = evaluate_at_threshold(y_test, test_probs, threshold=0.5)

print("\n=== Test (threshold=0.5) ===")
print(f"ROC AUC: {test_default['roc_auc']:.4f} | PR AUC: {test_default['pr_auc']:.4f} | "
      f"Precision: {test_default['precision']:.4f} | Recall: {test_default['recall']:.4f} | F1: {test_default['f1']:.4f}")
print("Confusion matrix (test):\n", test_default['confusion_matrix'])
print("Confusion matrix normalized (test):\n", np.round(test_default['confusion_matrix_normalized'], 3))

y_test_pred_default = (test_probs >= 0.5).astype(int)
print("\nClassification Report (test):\n", classification_report(y_test, y_test_pred_default, digits=4))

chosen_thr = float(best_thr_f1)
test_eval = evaluate_at_threshold(y_test, test_probs, threshold=chosen_thr)

print("\n=== Test (chosen F1-Max threshold from Validation evaluation) ===")
print(f"Threshold used: {chosen_thr:.3f}")
print(f"ROC AUC: {test_eval['roc_auc']:.4f} | PR AUC: {test_eval['pr_auc']:.4f} | "
      f"Precision: {test_eval['precision']:.4f} | Recall: {test_eval['recall']:.4f} | F1: {test_eval['f1']:.4f}")
print("Confusion matrix (test):\n", test_eval['confusion_matrix'])
print("Confusion matrix normalized (test):\n", np.round(test_eval['confusion_matrix_normalized'], 3))

y_test_pred = (test_probs >= chosen_thr).astype(int)
print("\nClassification Report (test):\n", classification_report(y_test, y_test_pred, digits=4))

fpr, tpr, roc_thresholds = roc_curve(y_test, test_probs)
roc_auc = roc_auc_score(y_test, test_probs)

precision, recall, pr_thresholds = precision_recall_curve(y_test, test_probs)
pr_auc = average_precision_score(y_test, test_probs)

prevalence = float(np.mean(y_test))

def _closest_idx(thrs, t):
    if thrs is None or len(thrs) == 0:
        return None
    return int(np.argmin(np.abs(thrs - t)))

mark_op_point = False
thr_idx_roc = thr_idx_pr = None
try:
    chosen_thr
    thr_idx_roc = _closest_idx(roc_thresholds, chosen_thr)
    idx_pr = _closest_idx(pr_thresholds, chosen_thr)
    thr_idx_pr = None if idx_pr is None else max(0, min(idx_pr + 1, len(precision) - 1))
    mark_op_point = True
except NameError:
    pass

fig, axes = plt.subplots(1, 2, figsize=figsize_double_column_split)

axes[0].plot(fpr, tpr, label=f'AUROC = {roc_auc:.3f}', linewidth=2, color=colors_blind_palette[0])
axes[0].plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='grey', label='Classificador aleatório')
if mark_op_point and thr_idx_roc is not None and 0 <= thr_idx_roc < len(tpr):
    axes[0].plot(fpr[thr_idx_roc], tpr[thr_idx_roc], marker='o', markersize=5, color='black', label=f'Ponto de corte ({chosen_thr:.2f})')
axes[0].set_xlabel('1 − Especificidade (FPR)')
axes[0].set_ylabel('Sensibilidade (TPR)')
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)
axes[0].grid(True, axis='both', linestyle='--', alpha=0.4)
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_linewidth(1.2)
axes[0].spines['bottom'].set_linewidth(1.2)
axes[0].legend(loc='lower right', frameon=False)
axes[0].text(0.01, 0.99, 'a', transform=axes[0].transAxes, fontsize=11, fontweight='bold', ha='left', va='bottom')

axes[1].plot(recall, precision, label=f'AUPRC = {pr_auc:.3f}', linewidth=2, color=colors_blind_palette[0])
axes[1].axhline(prevalence, linestyle='--', linewidth=1, color='grey', label=f'Prevalência (p={prevalence:.2f})')
if mark_op_point and thr_idx_pr is not None:
    axes[1].plot(recall[thr_idx_pr], precision[thr_idx_pr], marker='o', markersize=5, color='black', label=f'Ponto de corte ({chosen_thr:.2f})')
axes[1].set_xlabel('Revocação (Sensibilidade)')
axes[1].set_ylabel('Precisão')
axes[1].set_xlim(-0.02, 1.03)
axes[1].set_ylim(-0.02, 1.03)
axes[1].grid(True, axis='both', linestyle='--', alpha=0.4)
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_linewidth(1.2)
axes[1].spines['bottom'].set_linewidth(1.2)
axes[1].legend(loc='lower left', frameon=False)
axes[1].text(0.01, 0.99, 'b', transform=axes[1].transAxes, fontsize=11, fontweight='bold', ha='left', va='bottom')

plt.tight_layout()
plt.savefig('wdbc_roc_pr_test.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

y_test_pred = (test_probs >= chosen_thr).astype(int)
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.title('Matriz de confusão')
annotations = []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annotations.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
annotations = np.array(annotations).reshape(cm.shape)
sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
            xticklabels=['Benigno ', 'Maligno'],
            yticklabels=['Benigno ', 'Maligno'])
plt.savefig('matriz_confusao.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

plt.figure(figsize=(14, 6))
corr = df.drop('diagnosis', axis=1).corr()
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    annot_kws={"size": 7}
)
plt.title('Mapa de Calor de Correlação entre Features', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('mapa_calor.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()