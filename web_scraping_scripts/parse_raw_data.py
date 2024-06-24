import os
import pandas as pd
import re
from datetime import datetime

"""
Este script procesa archivos CSV de ventas y alquileres de una carpeta específica y 
guarda los datos en dos archivos pickle, uno para ventas y otro para alquileres. 

El script:
1. Determina la ubicación de la carpeta 'RawDataTFM' basada en la estructura del proyecto.
2. Lee todos los archivos CSV de ventas y alquileres dentro de esta carpeta.
3. Extrae la fecha del nombre del archivo si no está presente en los datos.
4. Guarda los datos procesados en archivos pickle dentro de la carpeta 'data'.
"""

class DataProcessor:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.data_folder = os.path.join(base_folder, 'data', 'RawDataTFM')
        self.venta_dfs = []
        self.alquiler_dfs = []

    def process_files(self):
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.data_folder, filename)
                df = pd.read_csv(file_path, delimiter=";")
                date = self.extract_date(filename)

                # Check if date is already in the data
                if 'fecha' not in df.columns:
                    df['fecha'] = date

                if 'venta' in filename:
                    self.venta_dfs.append(df)
                elif 'alquiler' in filename:
                    self.alquiler_dfs.append(df)

    def extract_date(self, filename):
        date_str = re.search(r'(\d{8})', filename).group(1)
        return datetime.strptime(date_str, '%Y%m%d').date()

    def save_pickles(self):
        data_save_folder = os.path.join(self.base_folder, 'data')
        if not os.path.exists(data_save_folder):
            os.makedirs(data_save_folder)

        venta_df = pd.concat(self.venta_dfs, ignore_index=True)
        alquiler_df = pd.concat(self.alquiler_dfs, ignore_index=True)

        venta_df.to_pickle(os.path.join(data_save_folder, 'ventas.pkl'))
        alquiler_df.to_pickle(os.path.join(data_save_folder, 'alquiler.pkl'))


def main():
    script_dir = os.path.dirname(__file__)
    base_folder = os.path.abspath(os.path.join(script_dir, '..'))
    processor = DataProcessor(base_folder)
    processor.process_files()
    processor.save_pickles()


if __name__ == "__main__":
    main()
