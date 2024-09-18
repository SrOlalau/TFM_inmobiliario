import streamlit as st
import streamlit.components.v1 as components
import os
from utils.assets_loader import set_assets, render_header, render_footer



# Definir la carpeta principal de gráficos
GRAPHICS_FOLDER = './static/graphs'

# Función para obtener archivos HTML en una carpeta específica
def get_html_files(graphics_folder):
    return [file for file in os.listdir(graphics_folder) if file.endswith('.html')]

# Función para clasificar gráficos por tipo
def classify_graphs(graph_files):
    graph_types = {
        'Histograma': [],
        'Correlación': [],
        'Boxplot': [],
        'Missing': [],
        'Scatter': [],
        'Otros': []
    }
    for graph in graph_files:
        if graph.startswith('histogram_'):
            graph_types['Histograma'].append(graph)
        elif graph.startswith('correlation_'):
            graph_types['Correlación'].append(graph)
        elif graph.startswith('boxplot_'):
            graph_types['Boxplot'].append(graph)
        elif graph.startswith('missing_'):
            graph_types['Missing'].append(graph)
        elif graph.startswith('scatter_'):
            graph_types['Scatter'].append(graph)
        else:
            graph_types['Otros'].append(graph)
    return graph_types


def render_descriptive_statistics():
    """
    Función para renderizar la tabla de estadísticas descriptivas desde un archivo HTML.
    """
    # Ruta al archivo HTML con la tabla de estadísticas descriptivas
    stats_html_path = os.path.join(GRAPHICS_FOLDER, 'descriptive_stats.html')

    # Lectura del contenido del archivo HTML
    with open(stats_html_path, 'r', encoding='utf-8') as file:
        stats_html_content = file.read()

    # Renderizar la tabla de estadísticas descriptivas en la aplicación Streamlit
    st.title("Tabla de Estadísticas Descriptivas")
    components.html(stats_html_content, height=600, scrolling=True)


# Aplicación Principal de Streamlit
def render_graphs():
    st.title("Estadísticas del Mercado Inmobiliario")
    st.markdown("""
        ### Estadísticas de distribución de las variables
        Explora gráficos y datos sobre las tendencias de precios en el mercado inmobiliario en distintas regiones de España.
    """)

    left_co, cent_co, right_co = st.columns(3)
    with left_co:
        # Selector de Alquiler o Venta
        transaction_type = st.radio("Selecciona el tipo de transacción:", ["Alquiler", "Venta"])

    # Obtener lista de gráficos disponibles en la carpeta seleccionada
    selected_folder = os.path.join(GRAPHICS_FOLDER, transaction_type.lower())
    available_graphs = get_html_files(selected_folder)

    # Clasificar gráficos por tipo
    classified_graphs = classify_graphs(available_graphs)
    with cent_co:
        # Dropdown para seleccionar tipo de gráfico
        graph_type = st.selectbox("Selecciona el tipo de gráfico:", list(classified_graphs.keys()))

    # Dropdown para seleccionar gráfico específico
    if graph_type and classified_graphs[graph_type]:
        with right_co:
            selected_graph = st.selectbox("Selecciona un gráfico para mostrar:", classified_graphs[graph_type])

        # Mostrar gráfico seleccionado
        if selected_graph:
            graph_html_path = os.path.join(selected_folder, selected_graph)
            with open(graph_html_path, 'r', encoding='utf-8') as file:
                graph_html_content = file.read()
            components.html(graph_html_content, height=500)
    else:
        st.write("No hay gráficos disponibles para esta selección.")





if __name__ == "__main__":
    st.set_page_config(page_title="Estadísticas", page_icon="📊", layout="wide")
    set_assets()
    # Mostrar graficos
    render_graphs()
    # Mostrar tabla de estadísticas descriptivas
    render_descriptive_statistics()

    render_footer()