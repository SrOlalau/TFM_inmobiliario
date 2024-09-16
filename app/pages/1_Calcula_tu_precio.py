import streamlit as st
import os
import pandas as pd
import pgeocode
from app.utils.data_loader import cargar_modelo  # Importar la función desde el módulo data_loader
from app.utils.assets_loader import set_assets, render_footer
from app.utils.data_processor import select_lat_lon
from app.utils.app_poi_calculator import POICalculator


# Definir las rutas de los modelos
MODEL_DIR = './machinelearning/models'
MODEL_ALQUILER_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_Alquiler.pickle')
MODEL_VENTA_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_Venta.pickle')

# Cargar los modelos
pipeline_alquiler, features_alquiler = cargar_modelo(MODEL_ALQUILER_PATH)
pipeline_venta, features_venta = cargar_modelo(MODEL_VENTA_PATH)


# Ruta del archivo de POIs
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POI_PATH = os.path.join(script_dir, 'data/POI/points_of_interest_ES.csv')

# Inicializar el calculador de POIs
poi_calculator = POICalculator(POI_PATH)

options_dict = {
    'mt2': 'metros cuadrados',
    'habitaciones': 'número de habitaciones',
    'banios': 'número de baños'
}


def obtener_variables_poi(features):
    """
    Extrae las variables relacionadas con POI desde las características del modelo.

    Args:
    features (dict): Diccionario de características del modelo.

    Returns:
    list: Lista de variables relacionadas con POI.
    """
    poi_vars = [var for var in features['options_range'].keys() if 'POI' in var or 'closest' in var]
    return poi_vars

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
                    user_input[col] = st.slider(f'Seleccione {options_dict[col]}:', min_value=int(rango80[0]),
                                                max_value=int(rango80[1]),
                                                value=int(default), step=1)
                # Dentro de la función mostrar_inputs
                elif col == 'mt2':
                    # Cambiar a un slider que permita seleccionar un rango
                    user_input[col] = st.slider(
                        f'Seleccione el rango de {options_dict[col]}:',
                        min_value=int(rango80[0]),
                        max_value=int(rango80[1]),
                        value=(int(default), int(default) + 10),  # Rango predeterminado
                        step=1
                    )

    # Establecer valores predeterminados para variables no incluidas en la lista filtrada
    for col in features['options_range']:
        if col not in filtered_vars:
            user_input[col] = features['options_range'][col]['default']

    # Asegurarse de que 'mes_publicado' esté configurado como None si no está en la lista filtrada
    user_input['mes_publicado'] = 12
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
    input_df1 = None
    input_df2 = None

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
            features = features_alquiler
            modelo = pipeline_alquiler
        elif modelo_seleccionado == 'Venta':
            st.subheader('Variables del Modelo de Venta')
            features = features_venta
            modelo = pipeline_venta

        # Obtener las variables relacionadas con POI
        poi_vars = obtener_variables_poi(features)

        # Calcular las métricas de POI usando la clase POICalculator
        poi_results = poi_calculator.calculate_point_estimations(lat, lon, poi_vars)


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

            # Incorporar las métricas de POI calculadas
            for var in poi_vars:
                # Si no existe usa default
                # input_usuario[var] = poi_results.get(var, features['options_range'][var]['default'])
                # Si no existe usa None
                input_usuario[var] = poi_results.get(var)

            pred_button = st.form_submit_button("Hacer Predicción")

        # Botón para hacer la predicción
        # Botón para hacer la predicción
        if pred_button:
            # Obtener los dos valores de mt2 seleccionados por el usuario
            mt2_min, mt2_max = input_usuario['mt2']

            # Crear dos entradas para el modelo, una para cada valor de mt2
            input_usuario_min = input_usuario.copy()
            input_usuario_min['mt2'] = mt2_min

            input_usuario_max = input_usuario.copy()
            input_usuario_max['mt2'] = mt2_max

            # Convertir las entradas del usuario en DataFrames para el modelo
            input_df1 = pd.DataFrame([input_usuario_min])
            input_df2 = pd.DataFrame([input_usuario_max])

        if input_df1 is not None and input_df2 is not None:
            # Realizar las predicciones para ambos DataFrames
            prediccion_min = modelo.predict(input_df1)
            prediccion_max = modelo.predict(input_df2)

            # Formatear los precios con separador de miles y símbolo de euros
            precio_min = f"{int(prediccion_min[0]):,}".replace(",", ".") + " €"
            precio_max = f"{int(prediccion_max[0]):,}".replace(",", ".") + " €"

            # Mostrar el resultado de las predicciones con tamaño de fuente más grande
            st.markdown(f'<h4>Resultado de la Predicción:</h3>',
                        unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-size:26px;">El valor estimado para {mt2_min} m² es: {precio_min}</h2>',
                        unsafe_allow_html=True)
            st.markdown(f'<h2 style="font-size:26px;">El valor estimado para {mt2_max} m² es: {precio_max}</h2>',
                        unsafe_allow_html=True)

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
