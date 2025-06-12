# Clasificador de Iris • FastAPI + Streamlit

Proyecto de demostración para cursos de Machine Learning y despliegue rápido de modelos:

* 🛠️ **Back-end** · FastAPI expone el modelo entrenado mediante dos rutas (`POST` y `GET`).
* 🖥️ **Front-end** · Streamlit llama a la API y muestra la predicción en una interfaz web.
* 📦 **ML** · Modelo de regresión logística entrenado con *scikit-learn* sobre el dataset Iris.

---

## 1 · Requisitos

| Herramienta | Versión mínima |
|-------------|----------------|
| Python      | 3.9 o superior |
| Git         | 2.25           |

Las librerías Python están listadas en `requirements.txt`.

---

## 2 · Instalación

```bash
# 1. Clona el repo
git clone https://github.com/tu-usuario/iris-fastapi-streamlit.git
cd iris-fastapi-streamlit

# 2. Crea y activa un entorno virtual
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (cmd)
.\.venv\Scriptsctivate

# 3. Instala las dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3 · Entrena el modelo (una sola vez)

```bash
python train_model.py
# → crea iris_logreg.joblib
```

---

## 4 · Ejecuta la API FastAPI

```bash
uvicorn app_post:app --reload
# Documentación interactiva en http://127.0.0.1:8000/docs
```

> **Rutas disponibles**
>
> * `POST /predict`   (body JSON)  
> * `GET  /predict`   ?data=5.1,3.5,1.4,0.2  
> * `GET  /predict_explicit`   ?sepal_len=…&sepal_wid=…&petal_len=…&petal_wid=…

---

## 5 · Ejecuta la interfaz Streamlit

En otra terminal (con el mismo entorno virtual activo):

```bash
streamlit run iris_streamlit.py
# Abrirá http://localhost:8501
```

---

## 6 · Pruebas rápidas con cURL

```bash
curl -X POST http://127.0.0.1:8000/predict      -H "accept: application/json"      -H "Content-Type: application/json"      -d "{"data":[5.1,3.5,1.4,0.2]}"
```

---

## 7 · Estructura del proyecto

```
.
├─ app_get.py            # API versión GET
├─ app_post.py           # API versión POST
├─ iris_streamlit.py     # Front-end Streamlit
├─ train_model.py        # Entrenamiento y serialización del modelo
├─ iris_logreg.joblib    # Modelo entrenado (generado)
├─ requirements.txt
└─ README.md
```

---

## 8 · Licencia

[MIT](LICENSE)

---

## 9 · Autor

Délany Ramírez — *Universidad Tecnológica de Pereira*  
Ideas, fallos o mejoras → abre un *issue* o *pull request*.
