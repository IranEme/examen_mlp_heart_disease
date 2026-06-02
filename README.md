# Clasificación de Enfermedad Cardíaca con Redes Neuronales (MLP)
Implementación de un pipeline de Inteligencia Artificial para la predicción de enfermedades cardíacas mediante un modelo de Perceptrón Multicapa.

**Alumno:** Juan Iran Lopez Mercado
**Matrícula:** [21760719]

## 1. Descripción del Dataset y Problema
El dataset Heart Disease proviene del UCI Machine Learning Repository y contiene información clínica de pacientes (edad, presión arterial, colesterol, etc.). El problema a resolver es de clasificación binaria: predecir la presencia (clase 1) o ausencia (clase 0) de enfermedad cardíaca.

## 2. Instrucciones de Instalación
Para replicar el entorno y las dependencias del proyecto, ejecuta:

pip install -r requirements.txt


## 3. Instrucciones de Ejecución
Ejecuta los siguientes scripts en el orden indicado:
1. python heart_disease_eda.py : Realiza el análisis exploratorio de datos y genera las visualizaciones.
2. python mlp_training.py : Preprocesa los datos, entrena la Red Neuronal, guarda el modelo serializado en la carpeta models/ y evalúa el rendimiento.

## 4. Análisis Exploratorio de Datos (EDA) - Hallazgos
* Balanceo de la clase objetivo: [Ej. El dataset presenta un balance adecuado, con 54% para la clase 0 y 46% para la clase 1].
* Histogramas numéricos: [Ej. Variables como el colesterol o la edad muestran una distribución que tiende a la normalidad, con algunos valores atípicos visibles].
* Boxplots por clase: [Ej. Existen diferencias claras en la distribución de ciertas variables, como la frecuencia cardíaca máxima (thalach), entre pacientes sanos y enfermos].
* Mapa de Correlación: [Ej. Las variables con mayor correlación con la clase objetivo son el tipo de dolor de pecho (cp) y la frecuencia cardíaca máxima (thalach)].


![(target: 0 vs 1)](screenshots/1.png)
![Histogramas de variables numéricas](screenshots/2.png)
![Countplots de variables categóricas](screenshots/2.1.png)
![Boxplots por clase objetivo)](screenshots/3.png)
![Mapa de correlación(Heatmap)](screenshots/4.png)


## 5. Arquitectura del Modelo
* Modelo: Perceptrón Multicapa (MLPClassifier)
* Capas ocultas: 2 capas (100 neuronas y 50 neuronas)
* Función de activación: ReLU
* Optimizador: Adam
* Parámetros adicionales: max_iter=500, early_stopping=True, random_state=42

## 6. Resultados del Entrenamiento (Métricas)
Tabla resumen de rendimiento sobre el conjunto de prueba (20%):

| Métrica   | Valor obtenido |
| :---      | :---           |
| Accuracy  | [Ej. 0.85]     |
| Precision | [Ej. 0.84]     |
| Recall    | [Ej. 0.86]     |
| F1-Score  | [Ej. 0.85]     |
| AUC       | [Ej. 0.90]     |

## 7. Análisis e Interpretación de Resultados
* Generalización (Overfitting/Underfitting): [Ej. El modelo generaliza correctamente. La curva de pérdida desciende de forma estable y las métricas en el conjunto de prueba son altas, lo que descarta un overfitting severo].
* Mejor clase predicha: [Ej. La clase X (0 o 1) se predice ligeramente mejor, como se observa en su F1-Score, debido a la distribución de las muestras en el entrenamiento].
* Variables más relevantes: De acuerdo con el análisis del EDA, variables como cp (tipo de dolor), thalach (frecuencia cardíaca máxima) y exang muestran un peso predictivo significativo.
* Propuesta de mejora: [Ej. Modificar los hiperparámetros como la tasa de aprendizaje, agregar regularización (alpha), o implementar técnicas de selección de características (Feature Selection) para eliminar variables con baja correlación].