import os
import glob
import pandas as pd
import re


def limpiar_numeros(numero_en_sucio):
    # Manejar valores NaN y convertir a cadena si es necesario
    if pd.isna(numero_en_sucio):
        return None
    # Convertir a cadena
    numero_en_sucio = str(numero_en_sucio)
    # Eliminar cualquier texto que no sea un número, punto, coma o espacio
    numero_en_sucio = re.sub(r'[^\d.,€\s]', '', numero_en_sucio)
    # Eliminar el símbolo de euro y espacios en blanco
    numero_en_sucio = re.sub(r'[€\s]', '', numero_en_sucio)
    # Reemplazar los puntos (separadores de miles) por nada
    numero_en_sucio = numero_en_sucio.replace('.', '')
    # Reemplazar las comas (separadores decimales) por nada
    numero_en_sucio = numero_en_sucio.replace(',', '')
    # Convertir a entero, manejar excepciones si la conversión falla
    try:
        return int(numero_en_sucio)
    except ValueError:
        return None


def merge_all_csv(script_dir):
    # Ruta a la carpeta que contiene los CSV's
    RawData_rstate = os.path.join(script_dir, 'data/RawDataTFM')
    OldData_rstate = os.path.join(script_dir, 'data/OldData')

    # Obtener todos los archivos .csv en el directorio RawData
    csv_files = glob.glob(os.path.join(RawData_rstate, '*.csv'))
    csv_olds = glob.glob(os.path.join(OldData_rstate, '*.csv'))

    # Lista para almacenar rutas completas de archivo
    file_paths = []

    # Iterar sobre los archivos y obtener las rutas completas
    for file in csv_files:
        file_paths.append(file)

    for file in csv_olds:
        file_paths.append(file)

    # Imprimir la lista de rutas completas de archivo
    print(file_paths)

    # Diccionario para almacenar los dataframes con sus nombres de archivo
    dataframes = {}

    # Iterar sobre cada archivo CSV y crear un dataframe
    for file in csv_files:
        # Extraer el nombre del archivo sin la ruta y la extensión
        file_name = file.split('/')[-1].replace('.csv', '')
        try:
            # Leer el archivo CSV en un dataframe de pandas
            df = pd.read_csv(file, delimiter=';', on_bad_lines='skip')

            # Determinar valor para columna 'fuente_datos'
            if 'trovit' in file_name:
                df['fuente_datos'] = 'trovit'
            else:
                df['fuente_datos'] = 'pisos.com'

            # Determinar valor para columna 'alquiler_venta'
            if 'alquiler' in file_name:
                df['alquiler_venta'] = 'alquiler'
            else:
                df['alquiler_venta'] = 'venta'

            # Determinar valor para columna 'fecha_extract'
            # Extraer la parte de la fecha del nombre del archivo
            fecha_extract_str = file_name.split('_')[-1]  # Obtener '20240630', '20240630', etc.
            # Convertir a formato de fecha
            fecha_extract = pd.to_datetime(fecha_extract_str, format='%Y%m%d')
            df['fecha_extract'] = fecha_extract

            # Almacenar el dataframe en el diccionario
            dataframes[file_name] = df

        except pd.errors.ParserError as e:
            print(f"Error parsing {file}: {e}")

    for file in csv_olds:
        # Extraer el nombre del archivo sin la ruta y la extensión
        file_name = file.split('/')[-1].replace('.csv', '')
        try:
            # Leer el archivo CSV en un dataframe de pandas
            df = pd.read_csv(file, delimiter=';', on_bad_lines='skip')

            df['fuente_datos'] = 'pisos.com'

            # Determinar valor para columna 'alquiler_venta'
            if 'alquiler' in file_name:
                df['alquiler_venta'] = 'alquiler'
            else:
                df['alquiler_venta'] = 'venta'

            # Determinar valor para columna 'fecha_extract'
            # Extraer la parte de la fecha del nombre del archivo
            fecha_extract_str = file_name.split('_')[-1]  # Obtener '20240630', '20240630', etc.
            # Convertir a formato de fecha
            fecha_extract = pd.to_datetime(fecha_extract_str, format='%Y%m%d')
            df['fecha_extract'] = fecha_extract

            # Almacenar el dataframe en el diccionario
            dataframes[file_name] = df

        except pd.errors.ParserError as e:
            print(f"Error parsing {file}: {e}")

    # Convertir el diccionario de dataframes en una lista de dataframes
    dfs = list(dataframes.values())

    # Concatenar todos los dataframes en uno solo
    consolidated_df = pd.concat(dfs, ignore_index=True)

    # Mostrar el dataframe consolidado
    print(consolidated_df.tail())

    # Aplicar la misma transformación al dataframe consolidado
    columns_to_clean_consolidated = ['precio', 'habitaciones', 'banios', 'mt2', 'publicado_hace', 'planta' ] # Asegúrate de ajustar según tus columnas reales
    for column in columns_to_clean_consolidated:
        consolidated_df[column] = consolidated_df[column].apply(limpiar_numeros)

    # Convertir las columnas 'latitude' y 'longitude' a tipo float
    consolidated_df['latitude'] = pd.to_numeric(consolidated_df['latitude'], errors='coerce')
    consolidated_df['longitude'] = pd.to_numeric(consolidated_df['longitude'], errors='coerce')
    dropped_df = consolidated_df.drop_duplicates(subset=consolidated_df.columns.difference(['fecha_extract']))

    output_path = os.path.join(script_dir, 'datamunging/consolidated_data_DM.csv')
    dropped_df.to_csv(output_path, index=False)


def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merge_all_csv(script_dir)


if __name__ == "__main__":
    main()