import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def data_tuning(script_dir):
    # Ruta del archivo CSV
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(file_path)
    #CAMBIO INCORPORADO DESDE EL EDA

    # Diccionario que mapea las ciudades a sus correspondientes CCAA en España
    ciudad_a_ccaa = {
    'sevilla': 'Andalucía',
    'Sevilla': 'Andalucía',  
    'malaga': 'Andalucía', 
    'Malaga': 'Andalucía',
    'cadiz': 'Andalucía', 
    'granada': 'Andalucía', 
    'Granada': 'Andalucía', 
    'Barcelona': 'Cataluña',
    'barcelona': 'Cataluña',
    'Cantabria': 'Cantabria',
    'cantabria': 'Cantabria', 
    'Madrid': 'Comunidad de Madrid', 
    'madrid': 'Comunidad de Madrid', 
    'valencia': 'Comunidad Valenciana', 
    'Valencia': 'Comunidad Valenciana', 
    'Alicante': 'Comunidad Valenciana',
    'Bizkaia': 'País Vasco', 
    'vizcaya_bizkaia': 'País Vasco'
    }

    # Mapear las ciudades a sus respectivas CCAA
    df['CCAA'] = df['CCAA'].map(ciudad_a_ccaa)
    
    #Sustituir Nan en plataforma por Pisos.com ya que era la única sin esa variable
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df[~df['mt2'].isin([0, np.inf, -np.inf]) & df['mt2'].notna()]
    df['plataforma']= df['plataforma'].replace(np.nan, 'Pisos')
    df = df.drop(columns=['publicado_hace','plataforma', 'raw_json', 'fuente_datos'])
    df['fecha_extract'] = pd.to_datetime(df['fecha_extract'], format='%Y-%m-%d')
    fecha_mas_antigua = df['fecha_extract'].min()
    df['mes_publicado'] = (df['fecha_extract'].dt.to_period('M') - fecha_mas_antigua.to_period('M')).apply(lambda x: x.n + 1)
    
    output_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df.to_csv(output_path, index=False)


def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_tuning(script_dir)

if __name__ == "__main__":
    main()