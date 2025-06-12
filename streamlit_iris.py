# iris_streamlit.py
import streamlit as st
import requests

# URL del endpoint FastAPI (la versión POST)
API_URL = "http://127.0.0.1:8000/predict"

st.title("Clasificador de Iris – demo Streamlit")
st.write(
    "Introduce las cuatro medidas y pulsa **Predecir** "
    "para consultar el modelo vía REST."
)

# ---------------------------------------------------------------------------
# Controles de entrada
# ---------------------------------------------------------------------------
sepal_len = st.number_input(
    "Largo del sépalo (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1
)
sepal_wid = st.number_input(
    "Ancho del sépalo (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1
)
petal_len = st.number_input(
    "Largo del pétalo (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1
)
petal_wid = st.number_input(
    "Ancho del pétalo (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1
)

# ---------------------------------------------------------------------------
# Llamada a la API
# ---------------------------------------------------------------------------
if st.button("Predecir"):
    payload = {
        "data": [sepal_len, sepal_wid, petal_len, petal_wid]
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API: {e}")
    else:
        result = resp.json()
        st.success(
            f"🌸 Predicción: **{result['class_name']}** "
            f"(etiqueta {result['prediction']})"
        )

## RUN IT
## streamlit run streamlit_iris.py