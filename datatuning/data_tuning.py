import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

def data_tuning(script_dir):
    # Ruta del archivo CSV
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(file_path)
    #Eliminar filas donde 'precio' es NaN, 0, o infinito
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df.drop(['planta', 'publicado_hace', 'raw_json'], axis=1)
    #CAMBIO INCORPORADO DESDE EL EDA
    # Se tramifica variable numérica y se vuelve de tipo objeto
    df['habitaciones'] = df['habitaciones'].apply(lambda x: '8 o más' if x >= 8 else x)
    df['banios'] = df['banios'].apply(lambda x: '7 o más' if x >= 7 else x)
    #CAMBIO INCORPORADO DESDE EL EDA
    #Homegeneizamos la comunidad autónoma
    df['CCAA'] = df['CCAA'].replace({
    'Malaga': 'malaga',
    'Sevilla': 'sevilla',
    'Granada': 'granada',
    'Madrid': 'madrid',
    'Barcelona': 'barcelona',
    'vizcaya_bizkaia': 'bizkaia',
    'Bizkaia': 'bizkaia',
    'Cantabria': 'cantabria',
    'Valencia': 'valencia',
    'Alicante': 'alicante'
    })
    #Reemplazar missings no declarados (ceros) por Nan
    num_var = df[['mt2', 'latitude', 'longitude']]
    df[num_var] = df[num_var].replace(0, np.nan)
    to_factor = list(df.loc[:,df.nunique() < 11])
    df[to_factor] = df[to_factor].astype('category')


    output_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df.to_csv(output_path, index=False)
    
def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_tuning(script_dir)

if __name__ == "__main__":
    main()