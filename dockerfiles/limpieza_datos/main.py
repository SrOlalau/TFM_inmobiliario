import os
import pandas as pd
import re
import psycopg2
from psycopg2 import sql
import requests
from datetime import datetime, timedelta

# Configura tu token de bot y el ID de chat
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

# Función para enviar mensajes a Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.get(url, params=params)
    return response

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

def fetch_data_from_db(conn, table_name, recent_days_filter=None):
    if recent_days_filter:
        query = f"SELECT * FROM {table_name} WHERE fecha >= CURRENT_DATE - INTERVAL '{recent_days_filter} days';"
    else:
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

def process_and_insert_data(df, source, target_conn, recent_days_filter=None):
    df['fuente_datos'] = source
    df['alquiler_venta'] = df['tipo']
    df['fecha_extract'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d', errors='coerce')

    # Asignar valor a la columna 'origen'
    if source == "trovit":
        df['origen'] = df['plataforma']
    else:
        df['origen'] = 'pisos.com'

    # Determinar las columnas a limpiar según la fuente de datos
    columns_to_clean = ['precio', 'habitaciones', 'banios', 'mt2']
    if source == "trovit":
        columns_to_clean.extend(['publicado_hace', 'planta'])
    elif source == "pisos.com":
        columns_to_clean.append('planta')

    # Asegurarse de convertir a texto las columnas numéricas
    for column in columns_to_clean:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: str(limpiar_numeros(x)) if pd.notna(x) else None)

    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Eliminar duplicados basándose en la columna href y fecha
    df = df.drop_duplicates(subset=['href', 'fecha'])

    # Asegurar que todas las columnas necesarias existan
    required_columns = ['precio', 'sub_descr', 'href', 'ubicacion', 'habitaciones', 'banios', 'mt2', 'otros',
                        'latitude', 'longitude', 'raw_json', 'ccaa', 'fuente_datos', 'alquiler_venta', 'fecha_extract',
                        'planta', 'publicado_hace', 'plataforma', 'origen']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    create_table_if_not_exists(target_conn)
    insert_or_update_data_into_db(target_conn, df)

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
                planta TEXT,
                publicado_hace TEXT,
                plataforma TEXT,
                origen TEXT,
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

def insert_or_update_data_into_db(conn, df):
    with conn.cursor() as cur:
        for index, row in df.iterrows():
            try:
                # Convertir los valores a texto si son numéricos y manejar None adecuadamente
                row_data = {}
                for col, val in row.items():
                    if val is None or pd.isna(val) or val == 'None':
                        row_data[col] = None
                    elif isinstance(val, (int, float)):
                        row_data[col] = str(val)
                    else:
                        row_data[col] = val

                int_fields = ['precio', 'habitaciones', 'banios', 'mt2']
                for field in int_fields:
                    if row_data[field] is not None:
                        try:
                            row_data[field] = int(float(row_data[field]))
                        except ValueError:
                            row_data[field] = None

                float_fields = ['latitude', 'longitude']
                for field in float_fields:
                    if row_data[field] is not None:
                        try:
                            row_data[field] = float(row_data[field])
                        except ValueError:
                            row_data[field] = None

                # Primero, verifica si el registro ya existe
                cur.execute(
                    "SELECT id FROM consolidated_data WHERE href = %s AND fecha_extract = %s",
                    (row_data['href'], row_data['fecha_extract'])
                )
                result = cur.fetchone()

                if result:
                    # Si existe, realiza un UPDATE
                    update_query = sql.SQL("""
                        UPDATE consolidated_data
                        SET precio = %s, sub_descr = %s, ubicacion = %s, habitaciones = %s, banios = %s, mt2 = %s,
                            planta = %s, publicado_hace = %s, plataforma = %s, origen = %s, otros = %s,
                            latitude = %s, longitude = %s, raw_json = %s, ccaa = %s, fuente_datos = %s, 
                            alquiler_venta = %s
                        WHERE id = %s
                    """)
                    cur.execute(update_query, (
                        row_data['precio'], row_data['sub_descr'], row_data['ubicacion'], row_data['habitaciones'], 
                        row_data['banios'], row_data['mt2'], row_data['planta'], row_data['publicado_hace'], 
                        row_data['plataforma'], row_data['origen'], row_data['otros'], row_data['latitude'], 
                        row_data['longitude'], row_data['raw_json'], row_data['ccaa'], row_data['fuente_datos'], 
                        row_data['alquiler_venta'], result[0]
                    ))
                else:
                    # Si no existe, realiza un INSERT
                    insert_query = sql.SQL("""
                        INSERT INTO consolidated_data (
                            precio, sub_descr, href, ubicacion, habitaciones, banios, mt2, planta,
                            publicado_hace, plataforma, origen, otros, latitude, longitude, raw_json,
                            ccaa, fuente_datos, alquiler_venta, fecha_extract
                        ) VALUES (
                            %(precio)s, %(sub_descr)s, %(href)s, %(ubicacion)s, %(habitaciones)s, %(banios)s,
                            %(mt2)s, %(planta)s, %(publicado_hace)s, %(plataforma)s, %(origen)s, %(otros)s,
                            %(latitude)s, %(longitude)s, %(raw_json)s, %(ccaa)s, %(fuente_datos)s,
                            %(alquiler_venta)s, %(fecha_extract)s
                        )
                    """)
                    cur.execute(insert_query, row_data)
                
            except Exception as e:
                print(f"Error inserting/updating row at index {index}: {row_data}")
                print(f"Exception: {e}")
                conn.rollback()
                raise e
        conn.commit()

def main():
    # Conexiones a las bases de datos de origen
    conn_pisos = connect_db("scraping_pisos", "pisos", "pisos", "10.1.2.2", "5437")
    conn_trovit = connect_db("scraping_trovit", "trovit", "trovit", "10.1.2.2", "5434")
    
    # Conexión a la base de datos de destino
    conn_target = connect_db("datos_limpios", "datos_limpios", "datos_limpios", "10.1.2.2", "5439")

    # Variable para activar o desactivar el filtro de 3 días
    use_recent_days_filter = False
    recent_days = 3

    try:
        # Enviar mensaje de inicio a Telegram
        send_telegram_message("Iniciando limpieza de datos")

        # Fetch and process data from pisos.com
        df_pisos = fetch_data_from_db(conn_pisos, "scraping_pisos_tabla", recent_days_filter=recent_days if use_recent_days_filter else None)
        process_and_insert_data(df_pisos, "pisos.com", conn_target)

        # Fetch and process data from trovit
        df_trovit = fetch_data_from_db(conn_trovit, "scraping_trovit_tabla", recent_days_filter=recent_days if use_recent_days_filter else None)
        process_and_insert_data(df_trovit, "trovit", conn_target)

        # Obtener el número total de filas en la base de datos de destino
        total_rows_query = "SELECT COUNT(*) FROM consolidated_data;"
        total_rows = pd.read_sql(total_rows_query, conn_target).iloc[0, 0]

        # Enviar mensaje de finalización a Telegram
        processed_rows = len(df_pisos) + len(df_trovit)
        send_telegram_message(f"Limpieza de datos finalizada. Filas procesadas: {processed_rows}. Filas totales en la base de datos: {total_rows}")

    finally:
        # Cerrar conexiones
        conn_pisos.close()
        conn_trovit.close()
        conn_target.close()

if __name__ == "__main__":
    main()
