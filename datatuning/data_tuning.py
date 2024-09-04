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
    #Sustituir Nan en plataforma por Pisos.com ya que era la única sin esa variable
    df['plataforma']= df['plataforma'].replace(np.nan, 'Pisos')
    output_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df.to_csv(output_path, index=False)


def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_tuning(script_dir)

if __name__ == "__main__":
    main()