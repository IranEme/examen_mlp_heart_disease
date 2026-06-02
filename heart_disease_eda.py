import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carga y Revisión Inicial del Dataset ---

# Como usamos la versión de UCI, definimos manualmente los nombres de las columnas
columnas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# 1. Cargar el dataset Heart Disease
df = pd.read_csv('data/heart.csv', names=columnas)

#El dataset de UCI usa el símbolo '?' para los valores faltantes. 
# Los reemplazamos por NaN (valores nulos oficiales de numpy) para que pandas los detecte.
df = df.replace('?', np.nan)

# Convertir el target a binario (0 = sano, 1 = con enfermedad cardíaca)
df['target'] = df['target'].apply(lambda x: 1 if float(x) > 0 else 0)

# 2. Mostrar las primeras 5 filas y las dimensiones
print("--- 2. Primeras 5 filas (.head) ---")
print(df.head())
print("\n--- Dimensiones del dataset (.shape) ---")
print(f"Filas, Columnas: {df.shape}")

# 3. Revisar los tipos de datos e info
print("\n--- 3. Tipos de datos (.dtypes) ---")
print(df.dtypes)
print("\n--- Información general (.info) ---")
df.info()

# 4. Verificar valores nulos o faltantes
print("\n--- 4. Valores nulos o faltantes ---")
print(df.isnull().sum())

# al correr esto, 'ca' y 'thal' tendrán nulos. 
# Aquí aplicamos una estrategia de imputación rápida llenando con la moda (el valor más frecuente),
# ya que son variables categóricas representadas con números.
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])
df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

# 5. Mostrar estadísticas descriptivas
print("\n--- 5. Estadísticas descriptivas (.describe) ---")
print(df.describe())

# --- 3.2 Análisis de la Variable Objetivo ---
print("\n--- 3.2 Distribución de la Clase Objetivo ---")
target_counts = df['target'].value_counts()
target_percentages = df['target'].value_counts(normalize=True) * 100

print(f"Conteo por clase:\n{target_counts}")
print(f"Porcentaje por clase:\n{target_percentages}")

# 1. Gráfica de distribución de la clase objetivo
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target', hue='target', palette='Set2', legend=False)
plt.title('Distribución de la Variable Objetivo (0 = Sano, 1 = Enfermo)')
plt.show()

# --- 3.3 Gráficas Obligatorias de Análisis ---
vars_num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# 2. Histogramas de variables numéricas
df[vars_num].hist(bins=15, figsize=(10, 8), color='teal', edgecolor='black')
plt.suptitle('Histogramas de Variables Numéricas')
plt.show()

# 3. Boxplots por clase objetivo
plt.figure(figsize=(12, 8))
for i, var in enumerate(vars_num, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='target', y=var, hue='target', data=df, palette='Set1', legend=False)
    plt.title(f'{var} vs Target')
plt.tight_layout()
plt.show()

# 4. Mapa de correlación (Heatmap)
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Correlación (Pearson)')
plt.show()

# Gráfica 4 obligatoria del Paso 3.3 (Opción 2 de la tabla): Countplots
vars_cat = ['cp', 'sex', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

plt.figure(figsize=(14, 10))
for i, var in enumerate(vars_cat, 1):
    plt.subplot(3, 3, i)
    # Usamos hue=var y legend=False para evitar advertencias de Seaborn en versiones nuevas
    sns.countplot(x=var, data=df, hue=var, palette='viridis', legend=False)
    plt.title(f'Frecuencia de {var}')
plt.tight_layout()
plt.show()
