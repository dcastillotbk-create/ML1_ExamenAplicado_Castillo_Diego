# ML1_ExamenAplicado_Castillo_Diego

Examen aplicado de Machine Learning I — Escuela de Ingeniería, Universidad Mayor.
**Autor:** Diego Castillo · **Fecha:** septiembre 2026

## Dataset

- **Nombre:** California Housing
- **Fuente:** scikit-learn (`sklearn.datasets.fetch_california_housing`), basado en el censo de EE.UU. de 1990 (Pace & Barry, 1997)
- **URL:** https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- **Filas:** 20.640 distritos censales
- **Columnas:** 8 predictoras numéricas continuas + 2 categóricas construidas (`age_group` ordinal, `region` nominal)
- **Variable objetivo:** `MedHouseVal` — valor medio de la vivienda del distrito, en cientos de miles de USD
- **Tipo de tarea:** Regresión

## Metodología

1. **EDA:** exploración inicial, análisis de nulos (0% en las 11 columnas), detección y capping de *outliers* por IQR, análisis de asimetría del objetivo (*skewness* = 0.98), correlaciones de Pearson y detección de multicolinealidad.
2. **Partición:** `train_test_split` 80/20 con `random_state=42`, aplicada **antes** de cualquier transformación del pipeline.
3. **Preprocesamiento:** `ColumnTransformer` con `SimpleImputer` (mediana / moda), `StandardScaler` para numéricas, `OrdinalEncoder` para `age_group` y `OneHotEncoder` para `region`. Ajustado exclusivamente sobre el conjunto de entrenamiento.
4. **Análisis no supervisado:** PCA (5 componentes, 86.40% de varianza explicada) y K-Means (K=2 óptimo según Silhouette Score).
5. **Modelado supervisado:** tres modelos con `GridSearchCV` (cv=5) y `random_state=42` — Ridge (penalizado), Random Forest (*bagging*) y Gradient Boosting (*boosting*).
6. **Evaluación:** RMSE, MAE, R² y MAPE sobre el conjunto de test, más análisis de importancia de variables y de las observaciones con mayor error.

## Resultados

| Modelo | RMSE | MAE | R² | MAPE | Brecha train-test | Tiempo (s) |
|---|---|---|---|---|---|---|
| **Gradient Boosting** ⭐ | **0.4720** | **0.3134** | **0.8300** | **18.21%** | **0.0727** | 321.4 |
| Random Forest | 0.5036 | 0.3274 | 0.8064 | 18.97% | 0.1678 | 1041.1 |
| Ridge | 0.6675 | 0.4881 | 0.6600 | 29.62% | 0.0175 | 0.2 |

**Modelo seleccionado: Gradient Boosting** — `n_estimators=200`, `max_depth=5`, `min_samples_split=5`.

Gana en las cuatro métricas, generaliza mejor que el Random Forest (brecha train-test de 0.07 contra 0.17) y además entrena tres veces más rápido, pese a construir los árboles en secuencia.

### Variables más importantes

| # | Variable | Importancia |
|---|---|---|
| 1 | `MedInc` | 56.2% |
| 2 | `AveOccup` | 13.3% |
| 3 | `Longitude` | 10.9% |
| 4 | `Latitude` | 9.9% |
| 5 | `HouseAge` | 4.3% |

## Hallazgo destacado

La variable objetivo está **censurada en 500.001 USD**: el censo trunca a ese valor todos los distritos más caros, afectando a **992 distritos (4.81% del dataset)**. Esto explica la mitad de los diez errores más grandes del modelo — en esos casos la predicción es razonable y lo que está recortado es el dato real, no la estimación.

## Contenido del repositorio

```
├── ML1_ExamenAplicado_Castillo_Diego.ipynb   # Notebook ejecutado
├── README.md
├── requirements.txt                          # dependencias del entorno de ejecución
├── resultados_modelos.csv                    # Tabla comparativa de modelos
└── figures/                                  # Todas las figuras (PNG, 150 dpi)
```

## Video

**Enlace:** https://drive.google.com/file/d/1wFOMYSpg2IHeWla07kKhZD7xeRKfYzah/view?usp=drive_link

Duración: 7:32. Recorrido por el notebook ejecutado siguiendo la estructura sugerida: dataset y justificación, EDA, PCA y K-Means, modelos y tabla comparativa, e interpretación y conclusiones.

## Cómo reproducir

```bash
pip install -r requirements.txt
jupyter notebook ML1_ExamenAplicado_Castillo_Diego.ipynb
```

El dataset se descarga automáticamente mediante `sklearn.datasets.fetch_california_housing`, por lo que no requiere configuración adicional ni acceso a Google Drive.

Las versiones de `requirements.txt` corresponden al entorno de Google Colab donde se ejecutó el notebook, filtradas a las librerías efectivamente utilizadas por el análisis.

> **Nota sobre tiempos de ejecución:** las dos celdas de `GridSearchCV` (Random Forest y Gradient Boosting) suman 100 ajustes de modelo y pueden tardar entre 10 y 20 minutos según el entorno.

## Declaración de uso de IA generativa

Se utilizó un asistente de IA (Claude, Anthropic) como apoyo para estructurar el pipeline de preprocesamiento, ordenar el código y revisar la redacción de las interpretaciones. La ejecución del análisis, la verificación de los resultados y las decisiones metodológicas fueron realizadas y validadas por el autor.
