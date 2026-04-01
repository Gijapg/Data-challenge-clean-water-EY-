# EY Water Quality Challenge

Proyecto de machine learning desarrollado para el **Challenge de Calidad del Agua de EY**, donde se buscaba mejorar un modelo baseline con métrica **R² = 0.203** mediante la predicción de tres parámetros de calidad del agua.

---

## Descripción del Challenge

### Baseline
- **Modelo Benchmark:** R² = 0.203
- **Objetivo:** Mejorar significativamente la métrica base
- **Evaluación:** Submission a través de archivo `submission.csv` evaluado en datos desconocidos

### Targets (3 modelos independientes)
1. **Total Alkalinity** (Alcalinidad Total)
2. **Electrical Conductance** (Conductancia Eléctrica)
3. **Dissolved Reactive Phosphorus** (Fósforo Reactivo Disuelto)

---

## Identificación del Desafío Principal

El análisis inicial reveló que **el desafío central era lograr que el modelo generalizara bien a nuevas locaciones geográficas**. Esto se debía a:

- Data de entrenamiento concentrada en regiones específicas
- Variabilidad geográfica alta en propiedades del agua
- Riesgo de overfitting a patrones locales

**Solución:** Rediseñar la estrategia de validación, ingeniería de features y selección de features.

---

## Estrategias Implementadas para Mejorar Generalización

### 1. Validación con GroupKFold

**Problema:** KFold estándar puede mezclar muestras del mismo río en train y validación.

**Solución:**
```python
river_group = Latitude.round(2) + '_' + Longitude.round(2)
GroupKFold(n_splits=5).split(X, groups=river_group)
```

**Beneficio:** Garantiza que el modelo valide en ríos/locaciones **nunca vistas** durante entrenamiento, evitando data leakage geográfico y proporcionando estimación realista de capacidad de generalización.

---

### 2. Features Geographicas sin Overfitting: K-Means Clustering

**Problema:** Usar coordenadas crudas (Lat/Lon) causa overfitting severo a las ubicaciones de entrenamiento.

**Solución: Clustering de Locaciones**

```python
# Paso 1: Crear 10 clusters geográficos naturales
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit([Latitude, Longitude])

# Paso 2: Para cada punto, calcular distancia a centroides
distances_to_centroids = [d_1, d_2, ..., d_10]

# Paso 3: Generar features
nearest_centroid_dist = min(distances_to_centroids)
second_nearest_dist = segunda_menor_distancia
spatial_isolation = nearest_dist / second_nearest_dist
```

**Por qué funciona:**
- Captura **información regional** sin memorizar coordenadas exactas
- Los 10 clusters representan cuencas/regiones naturales
- Distancias relativas generalizan a nuevas locaciones
- Feature importance detecta qué regiones son relevantes sin overfitting

**Comparación:**
```
Usar Lat/Lon crudo: modelo memoriza ubicaciones, falla en nuevas locaciones
Usar centros + distancias: modelo aprende patrones regionales, generaliza
```

---

### 3. PCA para Reducción de Dimensionalidad y Generalización

**Problema:** Features espectrales altamente correlacionadas (~0.7-0.9) generan colinealidad.

**Solución:**
```python
PCA(n_components=0.95)  # Retain 95% variance
# 6 features espectrales → 4 componentes PCA
```

**Beneficios:**
- Reduce ruido y overfitting
- Acelera entrenamiento
- Mejora estabilidad del modelo
- Generaliza mejor a nuevas distribuciones de datos

---

### 4. Extracción Masiva de Features + Feature Importance Filtering

**Proceso:**

1. **Generación:** Creación de ~100+ features candidatas:
   - Temporal: cyclical encoding (sin/cos), lags
   - Climáticas: ratios físicos, transformaciones log, interacciones
   - Espectrales: índices, ratios, combinaciones
   - Geoespaciales: K-means distances
   - Interacciones: climate × spectral

2. **Filtrado:** Selección con feature importance
   ```python
   # Obtener importancias de los modelos RandomForest
   importance = model.feature_importances_
   
   # Mantener features con importancia > threshold
   relevant_features = X[top_features]
   ```

3. **Validación:** Se probaron también métodos alternativos:
   - **SHAP values:** Explicabilidad de contribuciones por feature
   - **Permutation Importance:** Importancia basada en degradación de métrica
   - **Feature Importance (elegida):** Fue con las que se obtuvieron mejores resultados

**Resultado:** Dataset de ~75-80 features relevantes manteniendo interpretabilidad.

---

### 5. Ensemble de 20 Modelos con Semillas Distintas

**Objetivo:** Reducir varianza de predicción

```python
ENSEMBLE_SEEDS = [42, 52, 62, ..., 232]  # 20 semillas distintas

# Entrenar 20 RandomForest con diferentes semillas
predictions = [model_seed_42.predict(X), 
               model_seed_52.predict(X),
               ...,
               model_seed_232.predict(X)]

# Predicción final = promedio
y_pred_final = np.mean(predictions, axis=0)
```

**Beneficios:**
- Reduce varianza por inicialización aleatoria
- Proporciona estimación de incertidumbre (std de predicciones)
- Más robusto a nuevas locaciones que modelo único

---

## Arquitectura Final

```
Datos Crudos (9,319 muestras)
    ↓
Ingeniería de Features (~100+)
    ↓
PCA en Features Espectrales (6 → 4)
    ↓
Features Geográficas via K-Means (10 clusters)
    ↓
Feature Filtering con Feature Importance (~75-80 features)
    ↓
3 Modelos RandomForest (uno por target)
    ↓
Ensemble de 20 semillas por cada modelo
    ↓
Validación: GroupKFold (por locación)
    ↓
Predicción Final: /submission.csv
```

---

## Estructura del Proyecto

```
EY-Water-Challenge/
├── README.md                           # Este archivo
├── requirements.txt                    # Dependencias
├── GEE.env                             # Configuración Google Earth Engine
│
├── EDA_water_quality.py                # Análisis exploratorio
├── RF_challenge_water_prediction.py    # Modelo RandomForest principal
├── extraer_datos_nuevos.py             # Extractor de features GEE (opcional)
│
├── water_quality_training_dataset.csv  # 9,319 muestras × 3 targets
├── landsat_features_training.csv       # Bandas espectrales Landsat
├── climate_features_training.csv       # Variables climáticas PRISM
├── submission.csv                      # Predicciones finales
│
└── eda_outputs/                        # Reportes del análisis exploratorio
```

---

## Instalación y Uso

### Requisitos
- Python 3.8+

### Instalación
```bash
git clone https://github.com/tuuser/EY-Water-Challenge.git
cd EY-Water-Challenge
pip install -r requirements.txt
```

### Dependencias
```
numpy==1.26.4           # Operaciones numéricas y arrays
pandas==2.3.0           # Manipulación de dataframes
matplotlib==3.10.3      # Visualización de gráficos
scikit-learn==1.5.2     # Machine Learning (RandomForest, PCA, KMeans)
earthengine-api==1.7.14 # Acceso a datos satelitales de Google Earth Engine
```

### Ejecución

**1. Análisis Exploratorio de Datos (EDA):**
```bash
python EDA_water_quality.py
```

**Descripción:**
El script `EDA_water_quality.py` realiza un análisis exhaustivo del dataset:
- **Carga y fusión** de 3 CSVs (water_quality, landsat, climate) en un dataframe único
- **Detección de valores faltantes:** Reporta 3.33% de datos faltantes (principalmente en bandas Landsat)
- **Detección de outliers:** Utiliza método IQR para identificar anomalías por feature
- **Análisis de correlación:** Genera matriz de correlación y heatmaps para detectar relaciones entre features
- **Distribuciones de targets:** Visualiza histogramas de los 3 parámetros de calidad del agua
- **Perfiles temporales:** Analiza variación temporal (year/month) en los datos

**Salidas generadas en `eda_outputs/`:**
- `correlation_matrix_numeric.csv` - Matriz de correlaciones completa
- `missing_values_report.csv` - Detalle de valores faltantes por columna
- `outliers_iqr_report.csv` - Porcentaje de outliers detectados
- `target_correlation_report.csv` - Correlaciones de cada target con features
- `numeric_describe.csv` - Estadísticas descriptivas (media, std, min, max)
- `time_profile_year_month.csv` - Variación temporal de features

**De qué sirvió esta información:**
El EDA fue crucial para:
1. **Identificar features ruidosas:** Los outliers revelaron que coordenadas crudas causaban overfitting
2. **Detectar colinealidad:** Correlaciones 0.7-0.9 entre bandas espectrales justificaron usar PCA
3. **Confirmar falta de datos:** 3.33% de missingness indicó la necesidad de manejo robusto
4. **Entender distributions:** Observar que los targets eran aproximadamente normales sin sesgos extremos
5. **Guiar ingeniería de features:** Las correlaciones bajas iniciales motivaron crear 100+ nuevas features

---

**2. Entrenar Modelo y Generar Predicciones:**
```bash
python RF_challenge_water_prediction.py
# Genera: submission.csv
```

**Descripción:**
Script principal que implementa todas las estrategias descritas:
- Cargas datos y aplica ingeniería de features (K-Means, PCA, interacciones)
- Entrena 3 modelos RandomForest independientes (uno por target)
- Utiliza GroupKFold para validación geográfica
- Genera ensemble de 20 modelos con diferentes semillas
- Exporta predicciones a `submission.csv`

---

**3. (Opcional) Extraer Features desde Google Earth Engine:**

```bash
# Requiere cuenta GEE y autenticación
earthengine authenticate
python extraer_datos_nuevos.py
```

**Descripción del Script `extraer_datos_nuevos.py`:**

Este script automatiza la extracción de features satelitales de alta resolución usando Google Earth Engine API. Es útil para:
- **Enriquecer el dataset** con variables geofísicas no presentes en datos locales
- **Validación futura** con nuevas muestras
- **Feature engineering avanzado** basado en observaciones satelitales

**Pipeline de Extracción:**

1. **Inicialización de Google Earth Engine:**
   ```python
   load_env()  # Lee GEE_PROJECT_ID desde GEE.env
   init_earthengine(use_service_account=False)  # Autenticación interactiva
   ```
   Conecta con la API de Google Earth Engine usando credenciales seguras.

2. **Extracción Vectorizada de 9 Features Satelitales (no tienen que ser precisamente esas 9):**

3. **Combinación de Features con FeatureCollection Joins:**
   - Crea joins secuenciales para combinar todas las capas
   - Evita redundancia mediante inner joins
   - Mantiene integridad geoespacial con validación de geometrías

4. **Exportación a Google Drive:**
   - Generates CSV con 13 columnas: (id, latitude, longitude, date, + 9 features)
   - Archivo listo para integración al pipeline de ML

**¿De qué sirvió esta extracción?**
- **Aumento dimensionalidad:** De ~21 features iniciales a 30+ con variables satelitales
- **Captura de patrones regionales:** SMAP/MODIS reflejan humedad y vegetación del territorio
- **Validación de hipótesis:** Features como LAI y EVI correlacionan con calidad del agua
- **Generalización mejorada:** Variables satelitales son independientes de coordenadas crudas
- **Potencial futuro:** Permite hacer predicciones en nuevas locaciones con satelitales disponibles

**Nota de Seguridad:**
El proyecto ID de GEE se carga desde `GEE.env` (no hardcodeado), mantenito credenciales fuera del repositorio público.

---

## Resultados Obtenidos

| Métrica | Valor |
|---------|-------|
| **Baseline (Benchmark)** | R² = 0.203 |
| **Modelo Implementado** | R² = 0.382 |

**Nota:** La validación con GroupKFold proporciona estimación realista de desempeño en nuevas locaciones.

---

## � Configuración

### Archivo .env (GEE.env)
Para ejecutar el script de extracción de datos de Google Earth Engine, crea un archivo `GEE.env`:

```env
GEE_PROJECT_ID=tuproyecto-123456
```

---

## Decisiones Clave

| Decisión | Razón |
|----------|-------|
| **3 modelos independientes** | Cada target tiene patrones distintos |
| **GroupKFold validation** | Evalúa genuina capacidad de generalización geográfica |
| **K-Means + distancias** | Captura información regional sin memorizar coordenadas |
| **PCA en espectrales** | Reduce colinealidad y overfitting |
| **20 modelos ensemble** | Reduce varianza, mejora estabilidad |
| **Feature filtering** | Mantiene solo features con impacto real |

---

## Dataset

```
Muestras: 9,319
Features: 21 (después de fusión inicial)
          → 100+ (después de feature engineering)
          → 75-80 (después de filtrado por importancia)

Targets: 3 (TA, EC, DRP)
         
Cobertura geográfica: n_unique_locations = ~2,000+
Temporalidad: múltiples años

Valores faltantes: 3.33% (principalmente en bandas Landsat)
```

### Archivos de Entrada

| Archivo | Descripción | Muestras | Columnas |
|---------|-------------|----------|----------|
| `water_quality_training_dataset.csv` | Targets + metadatos base | 9,319 | id, latitude, longitude, date, + 3 targets |
| `landsat_features_training.csv` | Bandas espectrales Landsat 8 | 9,319 | ~12 bandas e índices espectrales |
| `climate_features_training.csv` | Variables climáticas PRISM | 9,319 | precipitación, temperatura, etc. |

### Archivos de Validación (Sin targets)

```
climate_features_validation.csv     # Datos climáticos para predicción
landsat_features_validation.csv     # Datos espectrales para predicción
submission_template.csv             # Template para envío de predicciones
```

---

## Flujo Completo de Ejecución

```
1. python EDA_water_quality.py
   ├─ Genera reportes en eda_outputs/
   └─ Identifica patrones y problemas en datos

2. python RF_challenge_water_prediction.py
   ├─ Lee datos preprocesados
   ├─ Ingeniería de features (~100+ features)
   ├─ Entrenamiento con GroupKFold (5 splits)
   ├─ Ensemble de 20 modelos por target
   └─ Genera submission.csv

3. (Opcional) python extraer_datos_nuevos.py
   ├─ Conecta a Google Earth Engine
   ├─ Extrae 9 features satelitales
   └─ Exporta a Google Drive (para futuras validaciones)
```

---

