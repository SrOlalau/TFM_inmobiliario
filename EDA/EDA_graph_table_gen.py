import os
import pandas as pd
import numpy as np
import plotly.express as px

# Ruta del archivo de datos y carpeta de gráficos
DATA_PATH = '../datamunging/consolidated_data.csv'
GRAPHICS_FOLDER = './graficos'

# Crear la carpeta de gráficos si no existe
os.makedirs(GRAPHICS_FOLDER, exist_ok=True)


def load_data(file_path):
    """Cargar datos desde un archivo CSV."""
    return pd.read_csv(file_path, low_memory=False)


def save_plot(fig, filename, folder=''):
    """Guardar gráfico en la carpeta especificada."""
    # Crear la ruta completa con la subcarpeta para alquiler o venta
    html_file_path = os.path.join(GRAPHICS_FOLDER, folder, filename)

    # Crear la carpeta si no existe
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)

    # Guardar el gráfico
    fig.write_html(html_file_path)
    print(f'Gráfico guardado en {html_file_path}')


def save_descriptive_stats(data, columns):
    """Guardar estadísticas descriptivas en formatos CSV y HTML con formato profesional."""
    # Generar estadísticas descriptivas para las columnas seleccionadas
    descriptive_stats = data[columns].describe(include='all').transpose()

    # Formatear la tabla: reemplazar NaN con cadenas vacías, redondear a 2 decimales y agregar separadores de miles
    formatted_stats = descriptive_stats.applymap(
        lambda x: '' if pd.isna(x) else '{:,.2f}'.format(x) if isinstance(x, (int, float)) else x)

    # Guardar como CSV
    csv_file_path = os.path.join(GRAPHICS_FOLDER, 'descriptive_stats.csv')
    formatted_stats.to_csv(csv_file_path)
    print(f'Estadísticas descriptivas guardadas en {csv_file_path}')

    # Guardar como HTML con estilo adicional para mejorar la presentación
    html_file_path = os.path.join(GRAPHICS_FOLDER, 'descriptive_stats.html')
    styled_stats = formatted_stats.style.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center'), ('background-color', '#f2f2f2'),
                                     ('padding', '5px')]},
        {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'right'), ('padding', '5px')]}
    ]).set_properties(**{'border': '1px solid black', 'border-collapse': 'collapse'})

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
    # Calcular los percentiles 2.5 y 97.5 para enfocarse en el 95% de los datos
    lower_bound, upper_bound = data[column].quantile([0.025, 0.975])

    # Crear el histograma con zoom en el 95% de los datos
    fig_histogram = px.histogram(
        data,
        x=column,
        title=f'Distribución de la variable {column}',
        marginal='box'  # Incluye un boxplot para visualizar los outliers
    )

    # Ajustar el rango del eje X para enfocar en el 95% de los datos
    fig_histogram.update_xaxes(range=[lower_bound, upper_bound])

    # Guardar el histograma
    save_plot(fig_histogram, f'histogram_{column}.html', folder)

    # Crear el gráfico de caja (boxplot)
    fig_boxplot = px.box(data, y=column, title=f'Boxplot de la variable {column}')
    save_plot(fig_boxplot, f'boxplot_{column}.html', folder)


def bivariate_analysis(data, col_x, col_y,folder):
    """Análisis bivariado entre dos columnas numéricas."""
    fig_scatter = px.scatter(data, x=col_x, y=col_y, title=f'Relación entre {col_x} y {col_y}')
    save_plot(fig_scatter, f'scatter_{col_x}_{col_y}.html',folder)


def correlation_heatmap(data,folder):
    """Generar un mapa de calor de la correlación entre variables numéricas."""
    # Filtrar solo columnas numéricas
    numeric_data = data.select_dtypes(include=np.number)

    # Calcular la matriz de correlación
    corr_matrix = numeric_data.corr()

    # Generar el gráfico de heatmap
    fig = px.imshow(corr_matrix, title='Mapa de Calor de Correlación')
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

    # Obtener los límites del 95% de la variable 'precio' para el ajuste del rango del eje X
    lower_bound, upper_bound = data['precio'].quantile([0.0, 0.9])

    # Filtrar las comunidades autónomas únicas
    unique_regions = data['CCAA'].dropna().unique()

    # Crear listas para almacenar los precios y etiquetas
    prices_by_region = []
    labels = []

    # Iterar sobre cada comunidad autónoma y extraer los precios correspondientes
    for region in unique_regions:
        regional_prices = data[data['CCAA'] == region]['precio'].dropna()
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
    save_plot(fig, 'price_distribution_by_region.html', folder)



def main():
    # Cargar datos
    data = load_data(DATA_PATH)

    print(data.loc[:,['latitude', 'longitude', 'POI_pharmacy_5km', 'POI_school_5km', 'closest_bus_station']].head())


    # # Filtrar columnas relevantes para estadísticas descriptivas
    # original_columns = [col for col in data.columns if "POI_" not in col and "closest_" not in col]
    #
    # # Guardar estadísticas descriptivas
    # save_descriptive_stats(data, original_columns)
    #
    # # Verificar los valores únicos en la columna 'alquiler_venta'
    # unique_values = data['alquiler_venta'].unique()
    # print(f"Valores únicos en 'alquiler_venta': {unique_values}")
    #
    # # Crear gráficos para cada categoría 'alquiler' y 'venta'
    # for category in ['alquiler', 'venta']:
    #     # Filtrar los datos por categoría
    #     category_data = data[data['alquiler_venta'] == category]
    #
    #     # Definir la subcarpeta para guardar los gráficos
    #     folder = category
    #
    #     # Visualizar datos faltantes para cada categoría
    #     visualize_missing_data(category_data, folder)
    #
    #     # Filtrar columnas numéricas que no contienen "POI_" o "closest_"
    #     numeric_columns = category_data.select_dtypes(include=np.number).columns
    #     numeric_columns = [col for col in numeric_columns if "POI_" not in col and "closest_" not in col]
    #
    #     # Análisis univariado por categoría
    #     for column in numeric_columns:
    #         univariate_analysis(category_data, column, folder)
    #
    #     # Análisis bivariado por categoría
    #     for i, col_x in enumerate(numeric_columns):
    #         for col_y in numeric_columns[i + 1:]:
    #             bivariate_analysis(category_data, col_x, col_y, folder)
    #
    #     # Mapa de calor de correlación por categoría
    #     correlation_heatmap(category_data[numeric_columns], folder)
    #
    #     # Gráfico de distribución de precios por CCAA
    #     price_distribution_by_region(category_data, folder)
    #
    #     # Manejo de outliers para la variable 'mt2'
    #     if 'mt2' in category_data.columns:
    #         category_data = handle_outliers(category_data, 'mt2', folder)
    #
    #



if __name__ == '__main__':
    main()
