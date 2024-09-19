import os
import folium as fl
import matplotlib as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import Normalize
from sqlalchemy import create_engine, text
from streamlit_folium import folium_static
import psycopg2
from utils.app_poi_calculator import POICalculator
from utils.assets_loader import set_assets, render_footer
from utils.data_loader import cargar_modelo  # Importar la función desde el módulo data_loader
from utils.data_processor import select_lat_lon
import math
from datetime import datetime, timedelta

st.set_page_config(page_title="Estima precios", page_icon="🗺️", layout="wide")

# Definir las rutas de los modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Obtiene el directorio base del script actual
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_ALQUILER_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_alquiler.pickle')
MODEL_VENTA_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_venta.pickle')

# Cargar los modelos
pipeline_alquiler, features_alquiler = cargar_modelo(MODEL_ALQUILER_PATH)
pipeline_venta, features_venta = cargar_modelo(MODEL_VENTA_PATH)



def cargar_pois_desde_db():

    """
    Carga los POIs desde una base de datos PostgreSQL.

    Returns:
    DataFrame: Un DataFrame con los datos de los POIs.
    """

    DB_NAME = "scraping_openstreetmaps"
    DB_USER = "POI"
    DB_PASSWORD = "POI"
    DB_HOST = "10.1.2.2"
    DB_PORT = "5438"
    DB_TABLE_NAME = "points_of_interest"
    try:
        # Establecer la conexión
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        
        # Ejecutar la consulta SQL
        query = f"SELECT * FROM {DB_TABLE_NAME}"
        pois_df = pd.read_sql(query, conn)
        
        # Cerrar la conexión
        conn.close()

        return pois_df

    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None
# Cargar los POIs desde la base de datos
poi_df = cargar_pois_desde_db()

# Inicializar el calculador de POIs
poi_calculator = POICalculator(poi_df)

options_dict = {
    'mt2': 'metros cuadrados',
    'habitaciones': 'número de habitaciones',
    'banios': 'número de baños'
}

# Parámetros de conexión a la base de datos
DB_NAME = "pred"
DB_USER = "pred"
DB_PASSWORD = "pred"
DB_HOST = "10.1.2.2"
DB_PORT = "5445"
DB_Table_name = "datos_finales_con_prediciones"

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
                        value=int(default),  # Rango predeterminado
                        step=1
                    )

    # Establecer valores predeterminados para variables no incluidas en la lista filtrada
    for col in features['options_range']:
        if col not in filtered_vars:
            user_input[col] = features['options_range'][col]['default']

    # Asegurarse de que 'mes_publicado' esté configurado como None si no está en la lista filtrada
    user_input['mes_publicado'] = 12
    return user_input


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

            pred_button = st.form_submit_button("Ver ofertas")

        if pred_button:
            mt2 = input_usuario['mt2']
            # Convertir las entradas del usuario en DataFrame para el modelo
            input_df = pd.DataFrame([input_usuario])

            if input_df is not None:
                # Realizar la predicción
                prediccion = modelo.predict(input_df)
                precio = f"{int(prediccion[0]):,}".replace(",", ".") + " €"
                st.markdown(f'<h2 style="font-size:26px;">El valor estimado para {mt2} m² es: {precio}</h2>',
                            unsafe_allow_html=True)

                # Cargar las ofertas cercanas basadas en la ubicación y los metros cuadrados
                df_filtrado = cargar_datos_filtrados(lat, lon, mt2, modelo_seleccionado)

                if df_filtrado.empty:
                    st.warning("No se encontraron ofertas cercanas.")
                else:
                    mostrar_mapa(lat, lon, df_filtrado)
            else:
                st.warning("Seleccione una ubicación para proceder con la predicción.")


def create_db_engine():
    engine_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url)
    return engine

def cargar_datos_filtrados(lat, lon, mt2, tipo):
    """
    Carga y filtra los registros de la base de datos según la ubicación y metros cuadrados,
    usando latitud y longitud máximas y mínimas para definir el área de búsqueda.

    Args:
    lat (float): Latitud del punto seleccionado.
    lon (float): Longitud del punto seleccionado.
    mt2 (int): Metros cuadrados seleccionados por el usuario.

    Returns:
    DataFrame: Un DataFrame con los registros filtrados por distancia y metros cuadrados.
    """
    try:
        # Definir el radio en kilómetros
        radio_km = 2

        # Cálculo de latitud y longitud máxima y mínima
        delta_lat = radio_km / 111.32
        delta_lon = radio_km / (111.32 * math.cos(math.radians(lat)))

        lat_min = lat - delta_lat
        lat_max = lat + delta_lat
        lon_min = lon - delta_lon
        lon_max = lon + delta_lon

        # Obtener la fecha actual y calcular la fecha límite (15 días atrás)
        fecha_limite = datetime.now() - timedelta(days=30)

        # Convertir la fecha límite en formato adecuado para SQL (YYYY-MM-DD HH:MM:SS)
        fecha_limite_str = fecha_limite.strftime('%Y-%m-%d')

        # Establecer la conexión a la base de datos de predicciones
        engine = create_db_engine()

        # Realizar la consulta con los límites de latitud y longitud calculados
        query = f"""
            SELECT * 
            FROM "{DB_Table_name}"
            WHERE ABS(CAST(mt2 AS FLOAT) - {mt2}) <= 15
            AND CAST(latitude AS FLOAT) BETWEEN {lat_min} AND {lat_max}
            AND CAST(longitude AS FLOAT) BETWEEN {lon_min} AND {lon_max}
            AND alquiler_venta = '{tipo.lower()}'
            AND DATE(fecha_extract) >= '{fecha_limite_str}'
        """

        df_filtrado = pd.read_sql(text(query), engine)
        engine.dispose()

        # Intentar convertir todas las columnas que puedan ser numéricas
        df_filtrado = df_filtrado.apply(pd.to_numeric, errors='coerce')
        df_filtrado = df_filtrado.drop_duplicates(subset=['latitude', 'longitude'], keep='last')

        return df_filtrado

    except Exception as e:
        st.error(f"Error al cargar datos filtrados: {e}")
        return pd.DataFrame()  # Devuelve un DataFrame vacío en caso de error


def mostrar_mapa(lat, lon, df_filtrado):
    """
    Muestra en un mapa los resultados filtrados y el punto seleccionado por el usuario.

    Args:
    lat (float): Latitud del punto seleccionado.
    lon (float): Longitud del punto seleccionado.
    df_filtrado (DataFrame): DataFrame con los registros cercanos.
    """
    norm = Normalize(vmin=df_filtrado['precio'].min(), vmax=df_filtrado['precio'].max())
    colormap = plt.colormaps['coolwarm']

    # Crear el mapa centrado en la ubicación seleccionada
    mapa = fl.Map(
        location=[lat, lon],
        tiles='https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://www.stamen.com/" target="_blank">Stamen Design</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        min_zoom=0,
        max_zoom=20,
        zoom_start=14
    )

    # Agregar el marcador del punto seleccionado
    fl.Marker([lat, lon], popup="Ubicación Seleccionada", icon=fl.Icon(color='blue')).add_to(mapa)

    # Agregar los marcadores de las ofertas cercanas
    if not df_filtrado.empty:
        for idx, row in df_filtrado.iterrows():
            color = colormap(norm(row['precio']))
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            popup_text = f"""
                            Precio: {row['precio']:,.0f} €,
                            mt2: {row['mt2']},
                            Habitaciones: {row['habitaciones']},
                            Baños: {row['banios']}
                            """
            if row['ratio'] > 0:
                circle_size = row['ratio']*80+10
            else:
                circle_size = 30
            fl.Circle(
                location=(row['latitude'], row['longitude']),
                radius=circle_size,
                color=hex_color,
                fill=True,
                fill_opacity=0.85,
                stroke=False,
                popup=popup_text,
            ).add_to(mapa)

    # Mostrar el mapa en la aplicación Streamlit
    folium_static(mapa)




if __name__ == "__main__":
    
    set_assets()
    # Crear interfaz en Streamlit
    st.title('Ofertas cercanas de Ventas y Alquiler')
    # Llamar a la función principal que genera las alternativas
    generar_alternativas()

    render_footer()
