import os
import pandas as pd
import numpy as np
import folium
from scipy.spatial import Voronoi
from streamlit_folium import folium_static
import streamlit as st
from utils.assets_loader import set_assets, render_footer
from utils.data_loader import cargar_modelo
import matplotlib as plt
from matplotlib.colors import Normalize
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from pyproj import Transformer
import branca.colormap as bcm
from sqlalchemy import create_engine, text

# Definir las rutas de los modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Obtiene el directorio base del script actual
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_ALQUILER_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_alquiler.pickle')
MODEL_VENTA_PATH = os.path.join(MODEL_DIR, 'random_forest_pipeline_alquiler_venta_venta.pickle')

# Parámetros de conexión a la base de datos
DB_NAME = "datatuning"
DB_USER = "datatuning"
DB_PASSWORD = "datatuning"
DB_HOST = "10.1.2.2"
DB_PORT = "5444"
DB_TABLE = "Datos_finales"

# Cargar los modelos
pipeline_alquiler, features_alquiler = cargar_modelo(MODEL_ALQUILER_PATH)
pipeline_venta, features_venta = cargar_modelo(MODEL_VENTA_PATH)


# Configuración de la página en Streamlit
st.set_page_config(page_title="Mapa de Calor de Precios", page_icon="🌍", layout="wide")
set_assets()
st.title('Mapa de Calor de Estimaciones de Precios en España')

# Definir especificaciones por comunidad autónoma
ccaa_dict = {
    'Comunidad de Madrid': {'center_lat': 40.436381, 'center_lon': -3.694906},
    'Cataluña': {'center_lat': 41.394600, 'center_lon': 2.157796},
    'Andalucía': {'center_lat': 37.386752, 'center_lon': -5.983399},
    'Cantabria': {'center_lat': 43.446667, 'center_lon': -3.820480},
    'Comunidad Valenciana': {'center_lat': 39.470616, 'center_lon': -0.376394},
    'País Vasco': {'center_lat': 43.263123, 'center_lon': -2.935009}
}


def cargar_datos_filtrados(selected_ccaa):
    """
    Conecta a la base de datos PostgreSQL y carga los datos filtrados por la comunidad autónoma seleccionada.
    """
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    query = text(f'SELECT * FROM "{DB_TABLE}" WHERE "ccaa" = :ccaa')
    df = pd.read_sql_query(query, engine, params={'ccaa': selected_ccaa})
    engine.dispose()

    # Eliminar registros sin latitud o longitud
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first')

    # Calcular la distancia desde el centro y mantener los puntos cercanos al centro
    center_lat = ccaa_dict[selected_ccaa]['center_lat']
    center_lon = ccaa_dict[selected_ccaa]['center_lon']
    df['distance_from_center'] = np.sqrt((df['latitude'] - center_lat) ** 2 + (df['longitude'] - center_lon) ** 2)

    return df


def preparar_entradas_para_modelo(df, mt2=50, habitaciones=1, banios=1):
    """
    Actualiza las entradas para el modelo de predicción basadas en los puntos de datos filtrados.
    """
    df['mt2'] = mt2
    df['habitaciones'] = habitaciones
    df['banios'] = banios
    df['mes_publicado'] = 12  # Valor fijo
    return df


def predecir_precio(df, modelo):
    """
    Realiza predicciones de precios para cada punto en el dataframe.
    """
    df['precio_estimado'] = modelo.predict(df)
    return df

def generar_voronoi_map(df, center_lat, center_lon, tasa=False):
    """
    Genera un mapa de calor usando diagramas de Voronoi.
    """

    perc_1 = df['precio_estimado'].quantile(0.01)
    perc_95 = df['precio_estimado'].quantile(0.99)
    df = df[(df['precio_estimado'] >= perc_1) & (df['precio_estimado'] <= perc_95)]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Crear el mapa centrado en la ubicación deseada con la capa base Stadia Stamen Toner Background
    mapa = folium.Map(
        location=[center_lat, center_lon],
        tiles='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=e0c70038-129c-47d1-a724-8bf0c7cc3e80',
        attr='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://www.stamen.com/" target="_blank">Stamen Design</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        min_zoom=0,
        max_zoom=20,
        zoom_start=13,
        width='100%',
        height='80%'
    )

    # Calcular el diagrama de Voronoi
    points = df[['longitude', 'latitude']].to_numpy()
    vor = Voronoi(points)

    norm = Normalize(vmin=df['precio_estimado'].min(), vmax=df['precio_estimado'].max())
    colormap = plt.colormaps['coolwarm']

    # Dibujar cada región de Voronoi
    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:  # Omitir regiones vacías o infinitas
            continue

        # Obtener los vértices del polígono de Voronoi en coordenadas (longitud, latitud)
        polygon_lonlat = [vor.vertices[i] for i in region]

        # Transformar coordenadas a sistema proyectado (metros)
        x_coords, y_coords = transformer.transform(
            [pt[0] for pt in polygon_lonlat],  # longitudes
            [pt[1] for pt in polygon_lonlat]  # latitudes
        )

        # Crear un polígono de shapely con las coordenadas proyectadas
        polygon_proj = Polygon(zip(x_coords, y_coords))

        # Verificar si el polígono es válido antes de calcular el área
        if not polygon_proj.is_valid:
            print(f"Polígono inválido: {explain_validity(polygon_proj)}")
            continue  # Salta este polígono si no es válido

        # Calcular el área en metros cuadrados
        area = polygon_proj.area

        tr_1 = 300*300
        tr_2 = 2200*2200
        start_op = 0.45
        end_op = 0.05
        # Calcular la opacidad basada en el área
        if area <= tr_1:
            fill_opacity = start_op
        elif (area > tr_1) & (area <= tr_2):
            fill_opacity = (start_op - end_op) * (tr_2 - area) / (tr_2 - tr_1)
        elif (area > tr_2) & (area <= tr_2*10):
            fill_opacity = end_op
        else:
            fill_opacity = 0.0

        # Obtener el color para la región
        price = df.iloc[point_idx]['precio_estimado']
        color = colormap(norm(price))
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

        # Convertir las coordenadas a formato (latitud, longitud) para Folium
        polygon_latlon = [[lat, lon] for lon, lat in polygon_lonlat]

        # Agregar el polígono al mapa con la opacidad ajustada
        folium.Polygon(
            locations=polygon_latlon,
            color=hex_color,
            fill=True,
            fill_opacity=fill_opacity,
            weight=0
        ).add_to(mapa)

    for _, row in df.iterrows():
        color = colormap(norm(row['precio_estimado']))
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        if tasa:
            popup_text = f"Tasa estimada: {row['precio_estimado'] * 100:.2f}%"
        else:
            popup_text = f"Precio estimado: {row['precio_estimado']:.2f}€"

        folium.Circle(
            location=(row['latitude'], row['longitude']),
            radius=30,  # Radio fijo de 25 metros
            color=hex_color,
            fill=True,
            fill_opacity=0.85,
            stroke=False,
            popup=popup_text
        ).add_to(mapa)

    # Crear un colorbar usando branca
    if tasa:
        # Convertir valores de 'precio_estimado' a porcentajes
        min_value = df['precio_estimado'].min() * 100
        max_value = df['precio_estimado'].max() * 100

        colorbar = bcm.LinearColormap(
            [colormap(norm(v)) for v in [df['precio_estimado'].min(), df['precio_estimado'].max()]],
            vmin=min_value,
            vmax=max_value,
            caption='Tasa estimada (%)'
        )
    else:
        colorbar = bcm.LinearColormap(
            [colormap(norm(v)) for v in [df['precio_estimado'].min(), df['precio_estimado'].max()]],
            vmin=df['precio_estimado'].min(),
            vmax=df['precio_estimado'].max(),
            caption='Precio Estimado (€)'
        )

    # Agregar el colorbar al mapa
    colorbar.add_to(mapa)

    return mapa

options_dict = {
    'mt2': 'metros cuadrados',
    'habitaciones': 'número de habitaciones',
    'banios': 'número de baños'
}
def main():
    model_co, ccaa_co = st.columns(2)
    with model_co:
        modelo_seleccionado = st.radio("Seleccione el modelo para realizar la predicción:", ('Alquiler', 'Venta'))
    with ccaa_co:
        selected_ccaa = st.selectbox('Seleccione la Comunidad Autónoma', list(ccaa_dict.keys()))

    # Agregar sliders para metros cuadrados, número de habitaciones y número de baños
    left_co, cent_co, last_co = st.columns(3)
    with left_co:
        mt2 = st.slider(f"Seleccione {options_dict['mt2']}", min_value=25, max_value=500, value=50, step=1)
    with cent_co:
        habitaciones = st.slider(f"Seleccione {options_dict['habitaciones']}", min_value=1, max_value=5, value=1, step=1)
    with last_co:
        banios = st.slider(f"Seleccione {options_dict['banios']}", min_value=1, max_value=5, value=1, step=1)

    if modelo_seleccionado == 'Alquiler':
        features = features_alquiler
        modelo = pipeline_alquiler
    else:
        features = features_venta
        modelo = pipeline_venta

    if st.button(f"Generar mapa de precios"):
        df_filtrado = cargar_datos_filtrados(selected_ccaa)
        # Preparar entradas para el modelo usando los valores seleccionados por el usuario
        df_preparado = preparar_entradas_para_modelo(df_filtrado, mt2=mt2, habitaciones=habitaciones, banios=banios)
        df_predicciones = predecir_precio(df_preparado.copy(), modelo)

        mapa_voronoi = generar_voronoi_map(df_predicciones, ccaa_dict[selected_ccaa]['center_lat'],
                                           ccaa_dict[selected_ccaa]['center_lon'])

        st.subheader(f'Mapa de calor de precios de {modelo_seleccionado}')
        folium_static(mapa_voronoi)

    if st.button("Generar mapa de tasas"):
        df_filtrado = cargar_datos_filtrados(selected_ccaa)
        # Preparar entradas para el modelo usando los valores seleccionados por el usuario
        df_preparado = preparar_entradas_para_modelo(df_filtrado, mt2=mt2, habitaciones=habitaciones, banios=banios)
        df_pred_alquiler = predecir_precio(df_preparado.copy(), pipeline_alquiler)
        df_pred_venta = predecir_precio(df_preparado.copy(), pipeline_venta)
        df_pred = df_preparado
        df_pred['precio_estimado'] = df_pred_alquiler['precio_estimado']*12/df_pred_venta['precio_estimado']

        mapa_voronoi = generar_voronoi_map(df_pred, ccaa_dict[selected_ccaa]['center_lat'],
                                           ccaa_dict[selected_ccaa]['center_lon'], tasa=True)

        st.subheader(f'Mapa de calor de Tasa anual: alquiler x 12 / venta')
        folium_static(mapa_voronoi)


if __name__ == "__main__":
    main()
    render_footer()
