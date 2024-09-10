import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, Column, Integer, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
import requests

# Configuración de las bases de datos
DB_SOURCE = {
    "NAME": "geo_y_poi",
    "USER": "geo_y_poi",
    "PASSWORD": "geo_y_poi",
    "HOST": "10.1.2.2",
    "PORT": "5442",
    "TABLE": "datos_limpios_con_geo_y_poi"
}

DB_DEST = {
    "NAME": "datatuning",
    "USER": "datatuning",
    "PASSWORD": "datatuning",
    "HOST": "10.1.2.2",
    "PORT": "5444",
    "TABLE": "Datos_finales"
}

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, json=payload)

def create_table_if_not_exists(engine_dest, engine_source, table_name):
    inspector = inspect(engine_dest)
    if not inspector.has_table(table_name):
        metadata = MetaData()
        metadata.reflect(bind=engine_source, only=[DB_SOURCE['TABLE']])
        source_table = metadata.tables[DB_SOURCE['TABLE']]

        new_table = Table(table_name, MetaData(),
                          Column('id', Integer, primary_key=True),
                          *[Column(c.name, c.type, primary_key=c.primary_key) 
                            for c in source_table.columns if c.name != 'id'])
        new_table.create(engine_dest)
        print(f"Tabla {table_name} creada.")
    else:
        print(f"Tabla {table_name} ya existe.")

def data_tuning():
    send_telegram_message("Iniciando proceso de datatuning.")

    engine_source = create_engine(f"postgresql://{DB_SOURCE['USER']}:{DB_SOURCE['PASSWORD']}@{DB_SOURCE['HOST']}:{DB_SOURCE['PORT']}/{DB_SOURCE['NAME']}")
    engine_dest = create_engine(f"postgresql://{DB_DEST['USER']}:{DB_DEST['PASSWORD']}@{DB_DEST['HOST']}:{DB_DEST['PORT']}/{DB_DEST['NAME']}")

    create_table_if_not_exists(engine_dest, engine_source, DB_DEST['TABLE'])

    df = pd.read_sql(f"SELECT * FROM {DB_SOURCE['TABLE']}", engine_source)
    filas_originales = len(df)
    print(f"Filas originales: {filas_originales}")

    df['ccaa'] = df['ccaa'].replace({
        'Malaga': 'malaga', 'Sevilla': 'sevilla', 'Granada': 'granada',
        'Madrid': 'madrid', 'Barcelona': 'barcelona', 'vizcaya_bizkaia': 'bizkaia',
        'Bizkaia': 'bizkaia', 'Cantabria': 'cantabria', 'Valencia': 'valencia',
        'Alicante': 'alicante'
    })

    # Filtrado de precios
    df_precio = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    filas_eliminadas_precio = len(df) - len(df_precio)
    print(f"Filas eliminadas por precio: {filas_eliminadas_precio}")

    # Filtrado de metros cuadrados
    df_final = df_precio[~df_precio['mt2'].isin([0, np.inf, -np.inf]) & df_precio['mt2'].notna()]
    filas_eliminadas_mt2 = len(df_precio) - len(df_final)
    print(f"Filas eliminadas por mt2: {filas_eliminadas_mt2}")

    df = df_final

    df['plataforma'] = df['plataforma'].fillna('Pisos')
    df = df.drop(columns=['publicado_hace','plataforma'])
    df['fecha_extract'] = pd.to_datetime(df['fecha_extract'], format='%Y-%m-%d')
    
    # Cambio en el cálculo de mes_publicado
    df['mes_publicado'] = df['fecha_extract'].dt.month

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Reemplazar la tabla completa
    df.to_sql(DB_DEST['TABLE'], engine_dest, if_exists='replace', index=False)

    filas_finales = len(df)
    
    resumen = f"""Resumen del proceso de datatuning:
    Filas en la tabla original: {filas_originales}
    Filas en la tabla final: {filas_finales}
    Filas eliminadas por precio inválido: {filas_eliminadas_precio}
    Filas eliminadas por mt2 inválido: {filas_eliminadas_mt2}
    Total de filas eliminadas: {filas_originales - filas_finales} """
    
    send_telegram_message(resumen)
    print(resumen)

def main():
    data_tuning()

if __name__ == "__main__":
    main()
