import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
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
        # Definir la estructura de la tabla basada en el DataFrame
        df = pd.read_sql(f"SELECT * FROM {DB_SOURCE['TABLE']} LIMIT 0", engine_source)
        df.to_sql(table_name, engine_dest, index=False, if_exists='fail')
        print(f"Tabla {table_name} creada.")
    else:
        print(f"Tabla {table_name} ya existe.")

def data_tuning():
    send_telegram_message("Iniciando proceso de actualización de datos.")

    # Conexión a la base de datos de origen
    engine_source = create_engine(f"postgresql://{DB_SOURCE['USER']}:{DB_SOURCE['PASSWORD']}@{DB_SOURCE['HOST']}:{DB_SOURCE['PORT']}/{DB_SOURCE['NAME']}")
    
    # Conexión a la base de datos de destino
    engine_dest = create_engine(f"postgresql://{DB_DEST['USER']}:{DB_DEST['PASSWORD']}@{DB_DEST['HOST']}:{DB_DEST['PORT']}/{DB_DEST['NAME']}")

    # Crear la tabla de destino si no existe
    create_table_if_not_exists(engine_dest, engine_source, DB_DEST['TABLE'])

    # Leer todos los datos de la base de datos de origen
    df = pd.read_sql(f"SELECT * FROM {DB_SOURCE['TABLE']}", engine_source)

    # Aplicar las transformaciones
    df['ccaa'] = df['ccaa'].replace({
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

    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df[~df['mt2'].isin([0, np.inf, -np.inf]) & df['mt2'].notna()]
    df['plataforma'] = df['plataforma'].replace(np.nan, 'Pisos')
    df = df.drop(columns=['publicado_hace','plataforma'])
    df['fecha_extract'] = pd.to_datetime(df['fecha_extract'], format='%Y-%m-%d')
    
    # Calcular mes_publicado
    fecha_mas_antigua = df['fecha_extract'].min()
    df['mes_publicado'] = (df['fecha_extract'].dt.to_period('M') - fecha_mas_antigua.to_period('M')).apply(lambda x: x.n + 1)

    # Reemplazar todos los datos en la tabla de destino
    df.to_sql(DB_DEST['TABLE'], engine_dest, if_exists='replace', index=False)

    total_rows = len(df)
    send_telegram_message(f"Proceso de actualización completado. Total de filas: {total_rows}")

def main():
    data_tuning()

if __name__ == "__main__":
    main()
