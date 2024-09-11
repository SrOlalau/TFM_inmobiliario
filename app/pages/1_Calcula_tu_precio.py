import streamlit as st
import os
import pandas as pd
import pgeocode
from app.utils.data_loader import cargar_modelo  # Importar la función desde el módulo data_loader
from app.utils.assets_loader import set_assets, render_footer
from app.utils.data_processor import select_lat_lon

# Definir las rutas de los modelos
MODEL_DIR = './app/models'
MODEL_ALQUILER_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_Alquiler.pickle')
MODEL_VENTA_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_Venta.pickle')

# Cargar los modelos
pipeline_alquiler, features_alquiler = cargar_modelo(MODEL_ALQUILER_PATH)
pipeline_venta, features_venta = cargar_modelo(MODEL_VENTA_PATH)


# Función para mostrar selectores de entrada basados en los rangos de características
def mostrar_inputs(features, variables_to_show=None):
    """
    Genera los controles de entrada en la interfaz basados en las características.

    Args:
    features (dict): Características y sus rangos de valores.
    variables_to_show (list, optional): Lista de variables a mostrar. Si es None, muestra todas.

    Returns:
    dict: Entradas del usuario.
    """
    user_input = {}

    # Determinar el orden de las variables: categóricas primero, luego numéricas
    categorical_vars = []
    numerical_vars = []

    # Identificar variables categóricas y numéricas en las características proporcionadas
    for col, info in features['options_range'].items():
        if isinstance(info['range'][0], str):
            categorical_vars.append(col)
        elif isinstance(info['range'][0], (int, float)):
            numerical_vars.append(col)

    # Crear una lista para mostrar, comenzando con variables categóricas
    if variables_to_show:
        filtered_vars = [var for var in variables_to_show if var in categorical_vars + numerical_vars]
    else:
        filtered_vars = categorical_vars + numerical_vars

    # Agregar las variables filtradas a la visualización, comenzando con las categóricas
    for col in filtered_vars:
        if col in features['options_range']:
            info = features['options_range'][col]
            rango = info['range']
            default = info['default']

            if isinstance(rango, list) and isinstance(rango[0], str):
                user_input[col] = st.selectbox(f'Seleccione {col}:', rango, index=rango.index(default))
            elif isinstance(rango, list) and isinstance(rango[0], (int, float)):
                rango80 = info['range_80pct']
                # Verificar si las columnas requieren pasos de enteros
                if col in ['banios', 'habitaciones']:
                    user_input[col] = st.slider(f'Seleccione {col}:', min_value=int(rango80[0]),
                                                max_value=int(rango80[1]),
                                                value=int(default), step=1)
                else:
                    user_input[col] = st.slider(f'Seleccione {col}:', min_value=rango80[0], max_value=rango80[1],
                                                value=default)

    # Establecer valores predeterminados para variables no incluidas en la lista filtrada
    for col in features['options_range']:
        if col not in filtered_vars:
            user_input[col] = features['options_range'][col]['default']

    # Asegurarse de que 'mes_publicado' esté configurado como None si no está en la lista filtrada
    user_input['mes_publicado'] = None
    return user_input


# Función para generar las alternativas de selección
def generar_alternativas():
    """
    Genera la interfaz de selección para el usuario y maneja las predicciones.

    Returns:
    None
    """
    # Llamada a la función para seleccionar la latitud y longitud
    location = select_lat_lon()

    input_features = ['mt2', 'habitaciones', 'banios']
    input_df = None

    # Verifica si la ubicación fue seleccionada
    if location:
        lat, lon, ccaa = location
        st.write(f"Ubicación seleccionada: Latitud {lat}, Longitud {lon}. {ccaa}")

        modelo_seleccionado = st.radio(
            "Seleccione el modelo para ver las variables:",
            ('Alquiler', 'Venta')
        )

        # Mostrar las variables del modelo seleccionado
        if modelo_seleccionado == 'Alquiler':
            st.subheader('Variables del Modelo de Alquiler')
            modelo = pipeline_alquiler
        elif modelo_seleccionado == 'Venta':
            st.subheader('Variables del Modelo de Venta')
            modelo = pipeline_venta


        with st.form(key='pred_form'):
            # Mostrar las variables del modelo seleccionado
            if modelo_seleccionado == 'Alquiler':
                input_usuario = mostrar_inputs(features_alquiler, variables_to_show=input_features)
            elif modelo_seleccionado == 'Venta':
                input_usuario = mostrar_inputs(features_venta, variables_to_show=input_features)

            # Actualiza las entradas del usuario con latitud y longitud seleccionadas
            input_usuario['latitude'] = lat
            input_usuario['longitude'] = lon
            input_usuario['CCAA'] = ccaa

            pred_button = st.form_submit_button("Hacer Predicción")

        # Botón para hacer la predicción
        if pred_button:
            # Convertir el input del usuario en un DataFrame para el modelo
            input_df = pd.DataFrame([input_usuario])

        if input_df is not None:
            # Realizar la predicción
            prediccion = modelo.predict(input_df)

            # Mostrar el resultado de la predicción
            st.subheader('Resultado de la Predicción:')
            st.write(f'El valor estimado es: {prediccion[0]:.2f}')
    else:
        st.warning("Seleccione una ubicación para proceder con la predicción.")


if __name__ == "__main__":
    st.set_page_config(page_title="Estima precios", page_icon="🖥️", layout="wide")
    set_assets()
    # Crear interfaz en Streamlit
    st.title('Estimación de Modelos de Ventas y Alquiler')
    # Llamar a la función principal que genera las alternativas
    generar_alternativas()

    render_footer()
