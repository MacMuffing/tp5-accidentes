# TP N° 5 - Predicción de gravedad de accidentes de tránsito

Trabajo práctico de entrenamiento de modelos de Machine Learning.

## Objetivo

Encontrar patrones en los accidentes de tránsito para ayudar a prevenir siniestros viales fatales y reducir la gravedad de los accidentes serios. Se entrena un modelo que estima la gravedad del accidente (Slight / Serious / Fatal) a partir de las condiciones previas: luz, clima, estado de la ruta, tipo de vía y zona urbana o rural.

## Dataset

- **Fuente:** [Accident Analysis Dataset (Kaggle)](https://www.kaggle.com/datasets/deeplumiere/accident-analysis-dataset)
- Accidentes de tránsito del Reino Unido, con más de 670.000 registros.
- En este repositorio se incluye solo una muestra de 20.000 filas (`muestra_datos.csv`) por el tamaño del archivo original.

## Modelos

- **Random Forest** (modelo principal)
- **Regresión Logística** (alternativa de comparación)

Ambos con `class_weight='balanced'` por el desbalance de clases (Fatal representa solo el 1,3% de los casos).

## Archivos

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación en Streamlit |
| `modelo_rf.joblib` | Modelo Random Forest entrenado |
| `columnas.joblib` | Columnas esperadas por el modelo |
| `muestra_datos.csv` | Muestra del dataset, sirve como ejemplo para la carga de archivos |
| `requirements.txt` | Librerías necesarias |
| `*.ipynb` | Notebook con el análisis exploratorio, limpieza y entrenamiento |

## Aplicación

La app permite ingresar un accidente individual o cargar un archivo CSV con varios casos y obtener la gravedad estimada.

**Acceso:** _(pegar acá la URL de Streamlit)_

## Ejecución local

```
pip install -r requirements.txt
streamlit run app.py
```
