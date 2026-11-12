import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción de gravedad de accidentes", layout="wide")

@st.cache_resource
def cargar():
    return joblib.load('modelo_rf.joblib'), joblib.load('columnas.joblib')

modelo, columnas = cargar()

FEATURES = ['Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions',
            'Road_Type', 'Urban_or_Rural_Area']

OPCIONES = {
    'Light_Conditions': ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                         'Darkness - lighting unknown', 'Darkness - lights unlit'],
    'Weather_Conditions': ['Fine no high winds', 'Raining no high winds', 'Other',
                           'Raining + high winds', 'Fine + high winds',
                           'Snowing no high winds', 'Fog or mist',
                           'Snowing + high winds', 'Desconocido'],
    'Road_Surface_Conditions': ['Dry', 'Wet or damp', 'Frost or ice', 'Snow',
                                'Flood over 3cm. deep', 'Desconocido'],
    'Road_Type': ['Single carriageway', 'Dual carriageway', 'Roundabout',
                  'One way street', 'Slip road', 'Desconocido'],
    'Urban_or_Rural_Area': ['Urban', 'Rural'],
}

def predecir(df):
    X = pd.get_dummies(df[FEATURES])
    X = X.reindex(columns=columnas, fill_value=0)
    pred = modelo.predict(X)
    prob = pd.DataFrame(modelo.predict_proba(X), columns=modelo.classes_)
    return pred, prob

st.title("Predicción de gravedad de accidentes de tránsito")
st.write("Modelo Random Forest entrenado con accidentes del Reino Unido. "
         "Estima la gravedad (Slight / Serious / Fatal) a partir de las condiciones del accidente.")

tab1, tab2 = st.tabs(["Un accidente", "Cargar archivo CSV"])

with tab1:
    c1, c2 = st.columns(2)
    entrada = {}
    for i, f in enumerate(FEATURES):
        with (c1 if i % 2 == 0 else c2):
            entrada[f] = st.selectbox(f, OPCIONES[f])
    if st.button("Predecir gravedad"):
        pred, prob = predecir(pd.DataFrame([entrada]))
        st.subheader(f"Gravedad estimada: {pred[0]}")
        st.bar_chart(prob.T.rename(columns={0: 'Probabilidad'}))

with tab2:
    st.write("Subí un CSV con las columnas: " + ", ".join(FEATURES))
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo:
        datos = pd.read_csv(archivo)
        faltan = [f for f in FEATURES if f not in datos.columns]
        if faltan:
            st.error(f"Faltan columnas: {faltan}")
        else:
            pred, prob = predecir(datos)
            datos['Gravedad_predicha'] = pred
            st.dataframe(datos)
            st.write(datos['Gravedad_predicha'].value_counts())
            st.download_button("Descargar resultados", datos.to_csv(index=False),
                               "resultados.csv", "text/csv")