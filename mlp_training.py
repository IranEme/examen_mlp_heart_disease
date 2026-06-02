import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, precision_recall_fscore_support)

# ==============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS )
# ==============================================================================
columnas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv('data/heart.csv', names=columnas)
df = df.replace('?', np.nan)

# Imputación de nulos detectados en el EDA
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])
df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

# Transformación a clasificación binaria estricta (0 vs 1)
df['target'] = df['target'].apply(lambda x: 1 if float(x) > 0 else 0)

# ==============================================================================
# 4.1 PREPROCESAMIENTO DE DATOS
# ==============================================================================
# 6. Separar features (X) y variable objetivo (y)
X = df.drop(columns=['target'])
y = df['target']

# 7. Dividir el dataset en entrenamiento (80%) y prueba (20%) con semilla 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 8. Aplicar normalización con StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Ajusta y transforma entrenamiento
X_test_scaled = scaler.transform(X_test)       # Aplica el mismo ajuste a prueba

# 9. Guardar el scaler ajustado
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# ==============================================================================
# 4.2 DEFINICIÓN Y ENTRENAMIENTO DEL MODELO MLP
# ==============================================================================
# Configuración con los parámetros mínimos obligatorios solicitados
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Dos capas ocultas con 100 y 50 neuronas
    activation='relu',            # Función de activación ReLU
    solver='adam',                # Optimizador Adam
    max_iter=500,                 # Máximo de épocas
    random_state=42,              # Semilla para reproducibilidad
    early_stopping=True           # Detención temprana si no mejora val_loss
)

# Entrenar el modelo
mlp.fit(X_train_scaled, y_train)

# ==============================================================================
# 4.3 EVALUACIÓN DEL MODELO - MÉTRICAS OBLIGATORIAS
# ==============================================================================
y_pred = mlp.predict(X_test_scaled)
y_prob = mlp.predict_proba(X_test_scaled)[:, 1] # Probabilidades para la curva ROC

print("\n" + "="*50)
print("             MÉTRICAS DE EVALUACIÓN DEL MODELO")
print("="*50)
print(f"Exactitud Global (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte completo de Clasificación:")
print(classification_report(y_test, y_pred))

# ==============================================================================
# 4.4 GRÁFICAS OBLIGATORIAS DEL MODELO (4 gráficas)
# ==============================================================================
# Gráfica 1: Matriz de Confusión (Heatmap)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sano (0)', 'Enfermo (1)'], 
            yticklabels=['Sano (0)', 'Enfermo (1)'])
plt.title('1. Matriz de Confusión (Heatmap)')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.tight_layout()
plt.show()

# Gráfica 2: Curva ROC con valor AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('2. Curva ROC')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Gráfica 3: Curva de pérdida (loss_curve_)
plt.figure(figsize=(6, 5))
plt.plot(mlp.loss_curve_, color='crimson', lw=2)
plt.title('3. Curva de Pérdida por Época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')
plt.tight_layout()
plt.show()

# Gráfica 4: Barplot de métricas por clase
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
df_metrics = pd.DataFrame({
    'Métrica': ['Precision', 'Recall', 'F1-Score'] * 2,
    'Valor': list(precision) + list(recall) + list(f1),
    'Clase': ['Clase 0 (Sano)'] * 3 + ['Clase 1 (Enfermo)'] * 3
})
plt.figure(figsize=(8, 5))
sns.barplot(x='Métrica', y='Valor', hue='Clase', data=df_metrics, palette='muted')
plt.ylim(0, 1.1)
plt.title('4. Comparación de Métricas por Clase')
plt.tight_layout()
plt.show()

# ==============================================================================
# 4.6 GUARDADO DEL MODELO ENTRENADO Y VALIDACIÓN
# ==============================================================================
# Guardar el modelo serializado
joblib.dump(mlp, 'models/mlp_heart_model.pkl')

# Validación requerida: Verificar existencia y mostrar tamaño en KB
print("\n" + "="*50)
print("             VALIDACIÓN DE ARCHIVOS GENERADOS")
print("="*50)
archivos_a_validar = ['models/mlp_heart_model.pkl', 'models/scaler.pkl']

for ruta in archivos_a_validar:
    if os.path.exists(ruta):
        tamano_kb = os.path.getsize(ruta) / 1024
        print(f"✔ El archivo '{ruta}' existe correctamente.")
        print(f"  Tamaño: {tamano_kb:.2f} KB\n")
    else:
        print(f"❌ ¡ERROR! No se encuentra el archivo en '{ruta}'.\n")