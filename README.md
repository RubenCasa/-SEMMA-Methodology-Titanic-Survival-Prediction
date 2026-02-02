# 🚢 SEMMA Methodology: Titanic Survival Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-green?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

## 📋 Descripción

Aplicación completa de la **metodología SEMMA** (Sample, Explore, Modify, Model, Assess) desarrollada por SAS Institute, utilizando el dataset Titanic de Kaggle para predecir la supervivencia de pasajeros.

Este proyecto demuestra un flujo de trabajo profesional de Data Science, desde la exploración inicial hasta la evaluación de múltiples modelos de Machine Learning.

---

## 🎯 Objetivo

Predecir qué pasajeros sobrevivieron al naufragio del RMS Titanic en 1912 utilizando técnicas de clasificación supervisada, aplicando rigurosamente las 5 fases de la metodología SEMMA.

---

## 📁 Estructura del Proyecto

```
PRO_ANALISIS/
├── 📂 data/
│   └── titanic.csv                    # Dataset original (891 registros)
├── 📂 notebooks/
│   └── SEMMA_Titanic_Completo.ipynb   # Notebook principal con todo el análisis
├── 📂 outputs/
│   ├── 📂 graficos/                   # 10 visualizaciones generadas
│   │   ├── 01_sample_distribution.png
│   │   ├── 02_explore_missing_values.png
│   │   ├── 03_explore_survival_analysis.png
│   │   ├── 04_explore_correlation_heatmap.png
│   │   ├── 05_explore_boxplots.png
│   │   ├── 06_modify_feature_engineering.png
│   │   ├── 07_assess_metrics_comparison.png
│   │   ├── 08_assess_confusion_matrices.png
│   │   ├── 09_assess_roc_curves.png
│   │   └── 10_assess_feature_importance.png
│   └── 📂 modelos/                    # Modelos entrenados (.pkl)
│       ├── logistic_regression_model.pkl
│       ├── random_forest_model.pkl
│       ├── xgboost_model.pkl
│       └── scaler.pkl
└── 📄 README.md
```

---

## 🔄 Metodología SEMMA

### 1️⃣ SAMPLE (Muestreo)
| Acción | Detalle |
|--------|---------|
| Carga de datos | 891 registros con 12 variables |
| División | 80% entrenamiento / 20% validación |
| Estratificación | Mantiene proporción de sobrevivientes en ambos conjuntos |

### 2️⃣ EXPLORE (Exploración)
| Análisis | Hallazgo |
|----------|----------|
| Valores faltantes | Age (~20%), Cabin (~77%), Embarked (0.2%) |
| Supervivencia por género | Mujeres: ~74% vs Hombres: ~19% |
| Supervivencia por clase | 1ra: ~63%, 2da: ~47%, 3ra: ~24% |

### 3️⃣ MODIFY (Modificación)
| Transformación | Descripción |
|----------------|-------------|
| Imputación Age | Mediana (28 años) |
| Imputación Embarked | Moda ('S' - Southampton) |
| FamilySize | SibSp + Parch + 1 |
| IsAlone | 1 si viaja solo, 0 si tiene familia |
| Title | Extraído del nombre (Mr, Mrs, Miss, Master, Rare) |
| HasCabin | 1 si tiene cabina asignada |

### 4️⃣ MODEL (Modelado)
| Modelo | Configuración |
|--------|---------------|
| **Logistic Regression** | Baseline, max_iter=1000 |
| **Random Forest** | GridSearchCV con optimización de hiperparámetros |
| **XGBoost** | 200 estimators, max_depth=5, learning_rate=0.1 |

### 5️⃣ ASSESS (Evaluación)

| Modelo               | Accuracy | AUC-ROC | CV Mean |
|----------------------|----------|---------|---------|
| Logistic Regression  | 0.8156   | 0.8581  | 0.7964  |
| Random Forest        | 0.7877   | 0.8411  | 0.8259  |
| XGBoost              | 0.8156   | 0.8418  | 0.8203  |

---

### 🏆Mejor Modelo
- **Modelo:** Logistic Regression  
- **AUC-ROC:** 0.8581  
- **Accuracy:** 0.8156

---

## 📊 Visualizaciones Generadas

<details>
<summary>📈 Click para ver las visualizaciones</summary>

### Fase SAMPLE
- **01_sample_distribution.png**: Distribución estratificada de clases (entrenamiento vs validación)

### Fase EXPLORE
- **02_explore_missing_values.png**: Análisis de valores faltantes por variable
- **03_explore_survival_analysis.png**: Supervivencia por género, clase y puerto
- **04_explore_correlation_heatmap.png**: Matriz de correlación entre variables numéricas
- **05_explore_boxplots.png**: Detección de outliers (Age, Fare, SibSp)

### Fase MODIFY
- **06_modify_feature_engineering.png**: Impacto de las nuevas variables creadas

### Fase ASSESS
- **07_assess_metrics_comparison.png**: Comparación de métricas entre modelos
- **08_assess_confusion_matrices.png**: Matrices de confusión de los 3 modelos
- **09_assess_roc_curves.png**: Curvas ROC comparativas
- **10_assess_feature_importance.png**: Importancia de variables (Random Forest)

</details>

---

## 🔍 Variables Más Importantes

Según el análisis de Random Forest:

1. **Title_encoded** - El título (Mr, Mrs, Miss) es el mejor predictor
2. **Sex_encoded** - El género sigue siendo crucial
3. **Fare** - El precio del boleto correlaciona con supervivencia
4. **Pclass** - La clase del pasajero
5. **Age** - La edad del pasajero

---

## 🚀 Ejecución

### Requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

### Ejecutar el Notebook
```bash
cd notebooks
jupyter notebook SEMMA_Titanic_Completo.ipynb
```

### Cargar Modelos Pre-entrenados
```python
import joblib

# Cargar modelo y scaler
model = joblib.load('outputs/modelos/random_forest_model.pkl')
scaler = joblib.load('outputs/modelos/scaler.pkl')

# Hacer predicciones
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

---

## 📚 Conclusiones

1. ✅ La metodología **SEMMA** proporciona un marco estructurado y reproducible para proyectos de ML.
2. ✅ El **feature engineering** (Title, FamilySize, IsAlone) mejoró significativamente los modelos.
3. ✅ Los resultados confirman patrones históricos: prioridad a mujeres, niños y clases altas.
4. ✅ Todos los modelos están listos para producción y exportados como archivos `.pkl`.

---

##  Autor

**Equipo Práctico de Ciencias de Datos  UNACH**

---

<p align="center">
  <i>⭐ Si este proyecto te fue útil, considera darle una estrella!</i>
</p>





