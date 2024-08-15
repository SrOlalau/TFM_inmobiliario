import os
import pandas as pd
import re
import psycopg2
from psycopg2 import sql

def connect_db(db_name, db_user, db_password, db_host, db_port):
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    return conn

def limpiar_numeros(numero_en_sucio):
    if pd.isna(numero_en_sucio):
        return None
    numero_en_sucio = str(numero_en_sucio)
    numero_en_sucio = re.sub(r'[^\d.,€\s]', '', numero_en_sucio)
    numero_en_sucio = re.sub(r'[€\s]', '', numero_en_sucio)
    numero_en_sucio = numero_en_sucio.replace('.', '')
    numero_en_sucio = numero_en_sucio.replace(',', '')
    try:
        return int(numero_en_sucio)
    except ValueError:
        return None

def fetch_data_from_db(conn, table_name):
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    return df

def validate_and_clean_data(df):
    max_int = 9223372036854775807  # Máximo para BIGINT
    min_int = -9223372036854775808

    for column in df.columns:
        if df[column].dtype == 'object':
            continue  # No necesitamos validar las columnas de texto
        if df[column].dtype in ['int64', 'float64']:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df[column] = df[column].apply(lambda x: x if pd.isna(x) or (min_int <= x <= max_int) else None)

    return df

def process_and_insert_data(df, source, target_conn):
    df['fuente_datos'] = source
    df['alquiler_venta'] = df['tipo']
    df['fecha_extract'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

    # Determinar las columnas a limpiar según la fuente de datos
    if source == "trovit":
        columns_to_clean = ['precio', 'habitaciones', 'banios', 'mt2', 'publicado_hace', 'planta']
    else:
        columns_to_clean = ['precio', 'habitaciones', 'banios', 'mt2', 'planta']

    # Asegurarse de convertir a texto las columnas numéricas
    for column in columns_to_clean:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: str(limpiar_numeros(x)) if pd.notna(x) else None)

    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    df = df.drop_duplicates(subset=df.columns.difference(['fecha_extract']))

    create_table_if_not_exists(target_conn)
    insert_data_into_db(target_conn, df)


def create_table_if_not_exists(conn):
    with conn.cursor() as cur:
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS consolidated_data (
                id SERIAL PRIMARY KEY,
                precio BIGINT,
                sub_descr TEXT,
                href TEXT,
                ubicacion TEXT,
                habitaciones INT,
                banios INT,
                mt2 BIGINT,
                otros TEXT,
                latitude FLOAT,
                longitude FLOAT,
                raw_json TEXT,
                ccaa TEXT,
                fuente_datos TEXT,
                alquiler_venta TEXT,
                fecha_extract DATE
            )
        """)
        cur.execute(create_table_query)
        conn.commit()

def clean_nan_values(df):
    df = df.fillna({
        'precio': 0,
        'habitaciones': 0,
        'banios': 0,
        'mt2': 0,
        'latitude': 0.0,
        'longitude': 0.0
    })
    return df

def ensure_correct_types(df):
    df['precio'] = df['precio'].astype('float64')
    df['habitaciones'] = df['habitaciones'].astype('float64')
    df['banios'] = df['banios'].astype('float64')
    df['mt2'] = df['mt2'].astype('float64')
    df['latitude'] = df['latitude'].astype('float64')
    df['longitude'] = df['longitude'].astype('float64')
    return df


def insert_data_into_db(conn, df):
    with conn.cursor() as cur:
        for index, row in df.iterrows():
            try:
                # Convertir los valores a texto si son numéricos y manejar None adecuadamente
                row_data = {}
                for col, val in row.items():
                    if val is None or val == "None":
                        row_data[col] = None
                    else:
                        row_data[col] = str(val) if isinstance(val, (int, float)) else val

                # Asegurarse de que 'latitude', 'longitude', y 'raw_json' están presentes en row_data
                if 'latitude' not in row_data:
                    row_data['latitude'] = None
                if 'longitude' not in row_data:
                    row_data['longitude'] = None
                if 'raw_json' not in row_data:
                    row_data['raw_json'] = None

                insert_query = sql.SQL("""
                    INSERT INTO consolidated_data (
                        precio, sub_descr, href, ubicacion, habitaciones, banios, mt2, otros,
                        latitude, longitude, raw_json, ccaa, fuente_datos, alquiler_venta, fecha_extract
                    ) VALUES (
                        %(precio)s, %(sub_descr)s, %(href)s, %(ubicacion)s, %(habitaciones)s, %(banios)s, %(mt2)s, %(otros)s,
                        %(latitude)s, %(longitude)s, %(raw_json)s, %(ccaa)s, %(fuente_datos)s, %(alquiler_venta)s, %(fecha_extract)s
                    )
                """)
                cur.execute(insert_query, row_data)
            except Exception as e:
                print(f"Error inserting row at index {index}: {row_data}")
                print(f"Exception: {e}")
                raise e  # Detener ejecución para investigar
        conn.commit()

def main():
    # Conexiones a las bases de datos de origen
    conn_pisos = connect_db("scraping_pisos", "pisos", "pisos", "10.1.2.2", "5437")
    conn_trovit = connect_db("scraping_trovit", "trovit", "trovit", "10.1.2.2", "5434")
    
    # Conexión a la base de datos de destino
    conn_target = connect_db("datos_limpios", "datos_limpios", "datos_limpios", "10.1.2.2", "5439")

    try:
        # Fetch and process data from pisos.com
        df_pisos = fetch_data_from_db(conn_pisos, "scraping_pisos_tabla")  # Aquí colocas el nombre correcto de la tabla para la base de datos de pisos
        process_and_insert_data(df_pisos, "pisos.com", conn_target)

        # Fetch and process data from trovit
        df_trovit = fetch_data_from_db(conn_trovit, "scraping_trovit_tabla")  # Aquí colocas el nombre correcto de la tabla para la base de datos de trovit
        process_and_insert_data(df_trovit, "trovit", conn_target)
    
    finally:
        # Cerrar conexiones
        conn_pisos.close()
        conn_trovit.close()
        conn_target.close()

if __name__ == "__main__":
    main()