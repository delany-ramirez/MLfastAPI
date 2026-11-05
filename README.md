# Clasificador de Iris • FastAPI + Streamlit

Proyecto de demostración para cursos de Machine Learning y despliegue rápido de modelos:

* 🛠️ **Back-end** · FastAPI expone el modelo entrenado mediante dos rutas (`POST` y `GET`).
* 🖥️ **Front-end** · Streamlit llama a la API y muestra la predicción en una interfaz web.
* 📦 **ML** · Modelo de regresión logística entrenado con *scikit-learn* sobre el dataset Iris.
* 📈 **MLOps** · MLflow registra cada entrenamiento y versiona el modelo que sirve la API.

---

## 1 · Requisitos

| Herramienta | Versión mínima |
|-------------|----------------|
| Python      | 3.12 o superior |
| uv          | 0.8            |
| Git         | 2.25           |

Las librerías Python están definidas en `pyproject.toml` y fijadas en `uv.lock`.

---

## 2 · Instalación

```bash
# 1. Clona el repo
git clone https://github.com/delany-ramirez/MLfastAPI.git
cd MLfastAPI

# 2. Crea el entorno virtual e instala las dependencias
uv sync
```

---

## 3 · Entrena el modelo

```bash
uv run train_model.py
# → registra una nueva versión de iris-logreg en MLflow (alias: champion)
```

Para revisar los experimentos y las versiones del modelo:

```bash
uv run mlflow ui
# Abrirá http://127.0.0.1:5000
```

Por defecto, MLflow guarda los datos en `mlflow.db` y `mlruns/`, dentro de la carpeta del proyecto. Ejecuta los comandos desde esa carpeta o define `MLFLOW_TRACKING_URI` para usar otro servidor.

La API carga el modelo al arrancar: si vuelves a entrenar, reiníciala para servir la nueva versión.

---

## 4 · Ejecuta la API FastAPI

```bash
uv run uvicorn app_post:app --reload
# Documentación interactiva en http://127.0.0.1:8000/docs
```

> **Rutas disponibles**
>
> * `app_post`: `POST /predict`   (body JSON)  
> * `app_get`: `GET  /predict`   ?data=5.1,3.5,1.4,0.2  
> * `app_get`: `GET  /predict_explicit`   ?sepal_len=…&sepal_wid=…&petal_len=…&petal_wid=…

---

## 5 · Ejecuta la interfaz Streamlit

En otra terminal:

```bash
uv run streamlit run streamlit_iris.py
# Abrirá http://localhost:8501
```

---

## 6 · Pruebas rápidas con cURL

```bash
curl -X POST http://127.0.0.1:8000/predict      -H "accept: application/json"      -H "Content-Type: application/json"      -d '{"data":[5.1,3.5,1.4,0.2]}'
```

---

## 7 · Estructura del proyecto

```
.
├─ app_get.py            # API versión GET
├─ app_post.py           # API versión POST
├─ streamlit_iris.py     # Front-end Streamlit
├─ train_model.py        # Entrenamiento y registro del modelo en MLflow
├─ model_registry.py     # Carga del modelo desde el registro de MLflow
├─ mlflow.db, mlruns/    # Datos de MLflow (generados)
├─ pyproject.toml
├─ uv.lock
└─ README.md
```

---

## 8 · Licencia

[MIT](LICENSE)

---

## 9 · Autor

Délany Ramírez — *Universidad Tecnológica de Pereira*  
Ideas, fallos o mejoras → abre un *issue* o *pull request*.
