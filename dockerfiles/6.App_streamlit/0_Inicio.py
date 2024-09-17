import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st
import requests
from sqlalchemy import create_engine
import plotly.figure_factory as ff
from utils.assets_loader import set_assets, render_header, render_footer

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    response = requests.post(url, data=data)
    return response.json()

# Configuración global de Plotly
pio.templates["custom_theme"] = pio.templates["plotly"]
pio.templates["custom_theme"].layout.font.family = "Roboto"
pio.templates["custom_theme"].layout.paper_bgcolor = "rgba(0, 0, 0, 0)"
pio.templates["custom_theme"].layout.plot_bgcolor = "rgba(0, 0, 0, 0)"
pio.templates["custom_theme"].layout.colorway = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
pio.templates.default = "custom_theme"

# Parámetros de conexión a la base de datos PostgreSQL
DB_DEST = {
    "NAME": "datatuning",
    "USER": "datatuning",
    "PASSWORD": "datatuning",
    "HOST": "10.1.2.2",
    "PORT": "5444",
    "TABLE": "Datos_finales"
}

# Especifica la ruta completa directamente usando os.path.join para compatibilidad multiplataforma
GRAPHICS_FOLDER = os.path.join('app', 'static', 'graphs')

NAMES = {
    'precio': 'Precio',
    'habitaciones': 'Habitaciones',
    'banios': 'Baños',
    'planta': 'Planta',
    'otros': 'Otros',
    'latitude': 'Latitud',
    'longitude': 'Longitud',
    'fecha_extract': 'Fecha de extracción',
    'plataforma': 'Plataforma de publicación'
}

STAT_NAMES = {
    'count': 'N°',
    'unique': 'N° valores únicos',
    'top': 'Valor más frecuente',
    'freq': 'Frecuencia',
    'mean': 'Prom.',
    'std': 'Desv. est.',
    'min': 'Mín.',
    'max': 'Máx.'
}

# Crear la carpeta de gráficos si no existe
os.makedirs(GRAPHICS_FOLDER, exist_ok=True)

def load_data(db_config):
    """Cargar datos desde una base de datos PostgreSQL."""
    try:
        # Crear la cadena de conexión
        connection_string = f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
        # Crear el motor de conexión
        engine = create_engine(connection_string)
        # Ajustar el nombre de la tabla según el caso
        query = f'SELECT * FROM "{db_config["TABLE"]}"'
        # Leer los datos de la tabla especificada
        data = pd.read_sql(query, engine)
        # Cerrar la conexión
        engine.dispose()
        return data
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        send_telegram_message(f"Error al conectar a la base de datos: {e}")
        return None

def save_plot(fig, filename, folder=''):
    """Guardar gráfico en la carpeta especificada con el tema y la fuente personalizados."""
    html_file_path = os.path.join(GRAPHICS_FOLDER, folder, filename)
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)

    font_import = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    </style>
    """

    fig_html = fig.to_html(
        full_html=True,
        config={'displayModeBar': False},
        include_plotlyjs='cdn',
        div_id='plotly-graph-div',
        default_height='100%',
        default_width='100%'
    )

    fig_html_with_font = fig_html.replace("<head>", f"<head>{font_import}")

    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(fig_html_with_font)

    print(f'Gráfico guardado en {html_file_path}')

def visualize_missing_data(data, folder):
    """Visualizar datos faltantes."""
    missing_data = data.isnull().mean() * 100
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    fig = px.bar(missing_data, x=missing_data.index, y=missing_data.values, title='Datos Faltantes en el Conjunto de Datos')
    save_plot(fig, 'missing_data.html', folder)

def univariate_analysis(data, column, folder):
    """Análisis univariado de una columna numérica."""
    column_name = column  # Se podría modificar para utilizar nombres personalizados
    lower_bound, upper_bound = data[column].quantile([0.025, 0.975])
    fig_histogram = px.histogram(
        data,
        x=column,
        title=f'Distribución de la variable {column_name}',
        marginal='box'
    )
    fig_histogram.update_xaxes(range=[lower_bound, upper_bound])
    save_plot(fig_histogram, f'histogram_{column}.html', folder)

    fig_boxplot = px.box(
        data,
        y=column,
        title=f'Boxplot de la variable {column_name}'
    )
    save_plot(fig_boxplot, f'boxplot_{column}.html', folder)

def bivariate_analysis(data, col_x, col_y, folder):
    """Análisis bivariado entre dos columnas numéricas."""
    col_x_name = col_x
    col_y_name = col_y

    fig_scatter = px.scatter(data, x=col_x, y=col_y,
                             title=f'Relación entre {col_x_name} y {col_y_name}')
    save_plot(fig_scatter, f'scatter_{col_x}_{col_y}.html', folder)

def correlation_heatmap(data, folder):
    """Generar un mapa de calor de la correlación entre variables numéricas."""
    numeric_data = data.select_dtypes(include=np.number)
    corr_matrix = numeric_data.corr()
    fig = px.imshow(corr_matrix, title='Mapa de Calor de Correlación', labels={'color': 'Correlación'})
    save_plot(fig, 'correlation_heatmap.html', folder)

def price_distribution_by_region(data, folder):
    """Crear un gráfico de distribución de la variable 'precio' agrupado por 'CCAA'."""
    for col in ['precio', 'Precio por mt2']:
        lower_bound, upper_bound = data[col].quantile([0.0, 0.9])
        unique_regions = data['ccaa'].dropna().unique()
        prices_by_region = []
        labels = []

        for region in unique_regions:
            regional_prices = data[data['ccaa'] == region][col].dropna()
            if not regional_prices.empty:
                prices_by_region.append(regional_prices.tolist())
                labels.append(region)

        fig = ff.create_distplot(prices_by_region, labels, show_hist=False, show_rug=False)
        fig.update_xaxes(range=[lower_bound, upper_bound])
        fig.update_layout(title_text='Distribución de Precios por Comunidad Autónoma')
        save_plot(fig, f'{col}_distribution_by_region.html', folder)

def check_and_generate_graphs():
    """Comprobar si los gráficos están generados, y si no, generarlos."""
    data = load_data(DB_DEST)

    if data is None:
        st.error("No se pudo cargar los datos desde la base de datos.")
        return

    data['Precio por mt2'] = data['precio'] / data['mt2']

    graph_exists = os.path.exists(os.path.join(GRAPHICS_FOLDER, 'venta', 'histogram_precio.html'))

    if graph_exists:
        st.write("Los gráficos ya están generados.")
    else:
        st.write("Generando gráficos, por favor espera...")
        send_telegram_message("Iniciando la generación de gráficos.")

        for category in ['alquiler', 'venta']:
            category_data = data[data['alquiler_venta'] == category]
            folder = category

            visualize_missing_data(category_data, folder)

            # Filtrar columnas numéricas que no contienen "POI_" o "closest_"
            numeric_columns = category_data.select_dtypes(include=np.number).columns
            numeric_columns = [col for col in numeric_columns if "poi_" not in col and "closest_" not in col]

            for column in numeric_columns:
                univariate_analysis(category_data, column, folder)

            for i, col_x in enumerate(numeric_columns):
                for col_y in numeric_columns[i + 1:]:
                    bivariate_analysis(category_data, col_x, col_y, folder)

            correlation_heatmap(category_data[numeric_columns], folder)
            price_distribution_by_region(category_data, folder)

        send_telegram_message("Los gráficos han sido generados.")
        st.write("Gráficos generados correctamente.")

def run_streamlit_app():
    send_telegram_message("La aplicación Streamlit está en línea.")
    st.set_page_config(page_title="Inicio", page_icon="👋", layout="wide")
    render_content()
    render_footer()

def render_content():
    st.title("Tu herramienta para predecir precios de viviendas en España")
    st.markdown("""
        ¡Gracias por visitar nuestra aplicación! Aquí podrás explorar estadísticas de precios,
        realizar predicciones de valor de inmuebles y encontrar ofertas vigentes en el mercado español.
    """)

    # Especifica la ruta al archivo HTML generado
    html_file_path = os.path.join(GRAPHICS_FOLDER, 'venta', 'histogram_precio.html')

    # Verificar si el archivo existe
    if os.path.exists(html_file_path):
        # Leer el contenido del archivo HTML
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Renderizar el HTML directamente en Streamlit
        st.components.v1.html(html_content, height=600)
    else:
        st.write("Los gráficos aún no están generados, se generarán después de la carga inicial.")

def main():
    run_streamlit_app()
    check_and_generate_graphs()

if __name__ == '__main__':
    main()
