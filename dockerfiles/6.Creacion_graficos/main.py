import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import psycopg2
import requests
from sqlalchemy import create_engine, text  # Nuevo: Para cargar datos desde PostgreSQL

# Configuración de la base de datos (Nuevo)
DB_DEST = {
    "NAME": "datatuning",
    "USER": "datatuning",
    "PASSWORD": "datatuning",
    "HOST": "10.1.2.2",
    "PORT": "5444",
    "TABLE": "Datos_finales"
}

# Configuración de Telegram (Nuevo)
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

# Función para enviar notificaciones por Telegram
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print('Mensaje enviado a Telegram con éxito.')
    else:
        print('Error al enviar mensaje a Telegram.')


# Define global Plotly layout settings
pio.templates["custom_theme"] = pio.templates["plotly"]
pio.templates["custom_theme"].layout.font.family = "Roboto"
pio.templates["custom_theme"].layout.paper_bgcolor = "rgba(0, 0, 0, 0)"  # Full translucent background
pio.templates["custom_theme"].layout.plot_bgcolor = "rgba(0, 0, 0, 0)"  # Full translucent background
pio.templates["custom_theme"].layout.colorway = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']  # Custom color palette
pio.templates.default = "custom_theme"


# Ruta del archivo de datos y carpeta de gráficos
def load_data_from_db():
    """Carga todos los datos desde PostgreSQL sin especificar columnas de fecha."""
    connection_string = f"postgresql://{DB_DEST['USER']}:{DB_DEST['PASSWORD']}@{DB_DEST['HOST']}:{DB_DEST['PORT']}/{DB_DEST['NAME']}"
    engine = create_engine(connection_string)

    # Cargar todos los datos de la tabla sin ningún filtro y sin procesar columnas de fecha
    query = text(f'SELECT * FROM "{DB_DEST["TABLE"]}"')
    data = pd.read_sql(query, engine)

    # No es necesario cerrar el connection_string, pero es recomendable cerrar el engine si no se utiliza más.
    engine.dispose()

    print(f"Tamaño del DataFrame cargado: {data.shape}")
    return data

GRAPHICS_FOLDER = '/resultado'

NAMES = {'precio': 'Precio',
         'habitaciones': 'Habitaciones',
         'banios': 'Baños',
         'planta': 'Planta',
         'otros': 'Otros',
         'latitude': 'Latitud',
         'longitude': 'Longitud',
         'fecha_extract': 'Fecha de extracción',
         'plataforma': 'Plataforma de publicación'
         }

STAT_NAMES = {'count': 'N°',
              'unique': 'N° valores únicos',
              'top': 'Valor más frecuente',
              'freq': 'Fecuencia',
              'mean': 'Prom.',
              'std': 'Desv. est.',
              'min': 'Mín.',
              'max': 'Máx.'

}

# Crear la carpeta de gráficos si no existe
os.makedirs(GRAPHICS_FOLDER, exist_ok=True)


def load_data(file_path):
    """Cargar datos desde un archivo CSV."""
    return pd.read_csv(file_path, low_memory=False)

def get_custom_name(original_name, name_dict):
    """Retorna el nombre personalizado de una variable o estadística si está en el diccionario; de lo contrario, devuelve el nombre original."""
    return name_dict.get(original_name, original_name)


def save_plot(fig, filename, folder=''):
    """Guardar gráfico en la carpeta especificada con el tema y la fuente personalizados."""
    # Crear la ruta completa con la subcarpeta para alquiler o venta
    html_file_path = os.path.join(GRAPHICS_FOLDER, folder, filename)

    # Crear la carpeta si no existe
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)

    # HTML custom header to include Roboto font
    font_import = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    </style>
    """

    # Export the figure as HTML with the custom font import
    fig_html = fig.to_html(
        full_html=True,           # Save the file as a full HTML document
        config={'displayModeBar': False}, # Hide toolbar
        include_plotlyjs='cdn',  # Use Plotly CDN for JavaScript
        div_id='plotly-graph-div',  # Use a consistent div ID
        default_height='100%',  # Set the default height to 100%
        default_width='100%'  # Set the default width to 100%
    )

    # Insert the font import directly after the <head> tag
    fig_html_with_font = fig_html.replace("<head>", f"<head>{font_import}")

    # Write the modified HTML content to the file
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(fig_html_with_font)

    print(f'Gráfico guardado en {html_file_path}')


def save_descriptive_stats(data, columns):
    """Guardar estadísticas descriptivas en formatos CSV y HTML con formato profesional."""
    # Generar estadísticas descriptivas para las columnas seleccionadas
    descriptive_stats = data[columns].describe(include='all').transpose()

    # Cambiar los nombres de las estadísticas según el diccionario STAT_NAMES
    descriptive_stats.rename(columns=lambda x: get_custom_name(x, STAT_NAMES), inplace=True)

    # Cambiar los nombres de las filas según el diccionario NAMES
    descriptive_stats.rename(index=lambda x: get_custom_name(x, NAMES), inplace=True)

    # Formatear la tabla: reemplazar NaN con cadenas vacías, redondear a 2 decimales y agregar separadores de miles
    formatted_stats = descriptive_stats.applymap(
        lambda x: '' if pd.isna(x) else '{:,.2f}'.format(x) if isinstance(x, (int, float)) else x)

    # Guardar como CSV
    csv_file_path = os.path.join(GRAPHICS_FOLDER, 'descriptive_stats.csv')
    formatted_stats.to_csv(csv_file_path)
    print(f'Estadísticas descriptivas guardadas en {csv_file_path}')

    # Guardar como HTML con estilo adicional para mejorar la presentación
    html_file_path = os.path.join(GRAPHICS_FOLDER, 'descriptive_stats.html')

    # Aplicar estilo personalizado
    styled_stats = formatted_stats.style.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center'), ('background-color', '#1BBED5'), ('color', 'white'), ('padding', '10px'), ('font-family', 'Roboto, sans-serif')]},
        {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'right'), ('padding', '8px'), ('font-family', 'Roboto, sans-serif')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#E4E4E4')]},
        {'selector': 'tbody tr:hover', 'props': [('background-color', '#f1f1f1')]},
        {'selector': 'thead th', 'props': [('border-bottom', '2px solid #ddd')]},
        {'selector': 'table', 'props': [('margin', '20px auto'), ('width', '90%'), ('border', '1px solid #ddd'), ('border-collapse', 'collapse'), ('font-family', 'Roboto, sans-serif')]}
    ])

    # Convertir a HTML con estilo
    styled_stats.to_html(html_file_path)
    print(f'Estadísticas descriptivas guardadas en {html_file_path}')


def visualize_missing_data(data,folder):
    """Visualizar datos faltantes."""
    missing_data = data.isnull().mean() * 100
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    fig = px.bar(missing_data, x=missing_data.index, y=missing_data.values, title='Datos Faltantes en el Conjunto de Datos')
    save_plot(fig, 'missing_data.html',folder)


def univariate_analysis(data, column, folder):
    """Análisis univariado de una columna numérica con enfoque en el 95% de los datos."""

    # Obtener el nombre personalizado de la columna
    column_name = get_custom_name(column, NAMES)

    # Calcular los percentiles 2.5 y 97.5 para enfocarse en el 95% de los datos
    lower_bound, upper_bound = data[column].quantile([0.025, 0.975])

    # Crear el histograma con zoom en el 95% de los datos
    fig_histogram = px.histogram(
        data,
        x=column,
        title=f'Distribución de la variable {column_name}',
        labels={column: column_name},  # Custom axis label
        marginal='box'  # Incluye un boxplot para visualizar los outliers
    )

    # Ajustar el rango del eje X para enfocar en el 95% de los datos
    fig_histogram.update_xaxes(range=[lower_bound, upper_bound])

    # Guardar el histograma
    save_plot(fig_histogram, f'histogram_{column}.html', folder)

    # Crear el gráfico de caja (boxplot)
    fig_boxplot = px.box(
        data,
        y=column,
        title=f'Boxplot de la variable {column_name}',
        labels={column: column_name}  # Custom axis label
    )
    save_plot(fig_boxplot, f'boxplot_{column}.html', folder)


def bivariate_analysis(data, col_x, col_y,folder):
    """Análisis bivariado entre dos columnas numéricas."""
    # Obtener nombres personalizados de las columnas
    col_x_name = get_custom_name(col_x, NAMES)
    col_y_name = get_custom_name(col_y, NAMES)

    fig_scatter = px.scatter(data,
                             x=col_x,
                             y=col_y,
                             title=f'Relación entre {col_x_name} y {col_y_name}',
                             labels={col_x: col_x_name, col_y: col_y_name}  # Custom axis labels
                             )
    save_plot(fig_scatter, f'scatter_{col_x}_{col_y}.html',folder)


def correlation_heatmap(data,folder):
    """Generar un mapa de calor de la correlación entre variables numéricas."""
    # Filtrar solo columnas numéricas
    numeric_data = data.select_dtypes(include=np.number)

    # Calcular la matriz de correlación
    corr_matrix = numeric_data.corr()

    # Get custom names for axis labels
    custom_axis_labels = {col: get_custom_name(col, NAMES) for col in corr_matrix.columns}

    # Generate the heatmap
    fig = px.imshow(
        corr_matrix,
        title='Mapa de Calor de Correlación',
        labels={'color': 'Correlation'},
        x=[custom_axis_labels[col] for col in corr_matrix.columns],  # Custom x-axis labels
        y=[custom_axis_labels[col] for col in corr_matrix.index]  # Custom y-axis labels
    )

    save_plot(fig, 'correlation_heatmap.html',folder)


def handle_outliers(data, column,folder):
    """Aplicar Winsorización para manejar outliers en una columna numérica."""
    q1, q3 = data[column].quantile([0.01, 0.99])
    data[column] = np.clip(data[column], q1, q3)  # Aplicación de Winsorización
    fig = px.box(data, y=column, title=f'Boxplot variable {column} tras Winsorización')
    save_plot(fig, f'boxplot_{column}_winsorized.html',folder)
    return data

import plotly.figure_factory as ff

def price_distribution_by_region(data, folder):
    """Crear un gráfico de distribución de la variable 'precio' agrupado por 'CCAA'."""

    for col in ['precio', 'Precio por mt2']:
        # Obtener los límites del 95% de la variable 'precio' para el ajuste del rango del eje X
        lower_bound, upper_bound = data[col].quantile([0.0, 0.9])

        # Filtrar las comunidades autónomas únicas
        unique_regions = data['CCAA'].dropna().unique()

        # Crear listas para almacenar los precios y etiquetas
        prices_by_region = []
        labels = []

        # Iterar sobre cada comunidad autónoma y extraer los precios correspondientes
        for region in unique_regions:
            regional_prices = data[data['CCAA'] == region][col].dropna()
            if not regional_prices.empty:  # Asegurar que no esté vacío
                prices_by_region.append(regional_prices.tolist())
                labels.append(region)

        # Crear el gráfico de distribución de densidad
        fig = ff.create_distplot(
            prices_by_region,
            labels,
            show_hist=False,   # No mostrar el histograma, solo la densidad
            show_rug=False     # No mostrar rug plot
        )

        # Limitar el rango del eje X al 95% de los datos
        fig.update_xaxes(range=[lower_bound, upper_bound])

        # Título del gráfico
        fig.update_layout(title_text='Distribución de Precios por Comunidad Autónoma')

        # Guardar el gráfico
        save_plot(fig, f'{col}_distribution_by_region.html', folder)



def main():
    # Notificación de inicio
    send_telegram_message('El proceso ha comenzado.')

    # Cargar datos desde la base de datos PostgreSQL
    data = load_data_from_db()

    data['Precio por mt2'] = data['precio']/data['mt2']

    # Filtrar columnas relevantes para estadísticas descriptivas
    original_columns = [col for col in data.columns if "POI_" not in col and "closest_" not in col]

    # Guardar estadísticas descriptivas
    save_descriptive_stats(data, original_columns)

    # Verificar los valores únicos en la columna 'alquiler_venta'
    unique_values = data['alquiler_venta'].unique()
    print(f"Valores únicos en 'alquiler_venta': {unique_values}")

    # Crear gráficos para cada categoría 'alquiler' y 'venta'
    for category in ['alquiler', 'venta']:
        # Filtrar los datos por categoría
        category_data = data[data['alquiler_venta'] == category]

        # Definir la subcarpeta para guardar los gráficos
        folder = category

        # Visualizar datos faltantes para cada categoría
        visualize_missing_data(category_data, folder)

        # Filtrar columnas numéricas que no contienen "POI_" o "closest_"
        numeric_columns = category_data.select_dtypes(include=np.number).columns
        numeric_columns = [col for col in numeric_columns if "POI_" not in col and "closest_" not in col]

        # Análisis univariado por categoría
        for column in numeric_columns:
            univariate_analysis(category_data, column, folder)

        # Análisis bivariado por categoría
        for i, col_x in enumerate(numeric_columns):
            for col_y in numeric_columns[i + 1:]:
                bivariate_analysis(category_data, col_x, col_y, folder)

        # Mapa de calor de correlación por categoría
        correlation_heatmap(category_data[numeric_columns], folder)

        # Gráfico de distribución de precios por CCAA
        price_distribution_by_region(category_data, folder)

        # Manejo de outliers para la variable 'mt2'
        if 'mt2' in category_data.columns:
            category_data = handle_outliers(category_data, 'mt2', folder)

    # Notificación de finalización
    send_telegram_message('El proceso ha finalizado.')

if __name__ == '__main__':
    main()
