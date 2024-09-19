import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import optuna
import gc
import requests

warnings.filterwarnings('ignore')

# Configuración de la base de datos
DB_DEST = {
    "NAME": "datatuning",
    "USER": "datatuning",
    "PASSWORD": "datatuning",
    "HOST": "10.1.2.2",
    "PORT": "5444",
    "TABLE": "Datos_finales"
}

DB_PRED = {
    "NAME": "pred",
    "USER": "pred",
    "PASSWORD": "pred",
    "HOST": "10.1.2.2",
    "PORT": "5445",
    "TABLE": "datos_finales_con_prediciones"
}

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

# Directorio donde están los pickles
PICKLE_DIR = "/resultados"

def create_db_engine(db_config):
    engine_url = f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
    engine = create_engine(engine_url)
    return engine

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print("Mensaje enviado a Telegram.")
    except Exception as e:
        print(f"Error al enviar mensaje a Telegram: {str(e)}")

def load_data_from_postgres():
    connection_string = f"postgresql://{DB_DEST['USER']}:{DB_DEST['PASSWORD']}@{DB_DEST['HOST']}:{DB_DEST['PORT']}/{DB_DEST['NAME']}"
    engine = create_engine(connection_string)
    query = text(f'SELECT * FROM "{DB_DEST["TABLE"]}"')
    df = pd.read_sql(query, engine)
    print(f"Tamaño del DataFrame cargado: {df.shape}")
    return df

def load_models():
    venta_model_path = os.path.join(PICKLE_DIR, 'random_forest_pipeline_alquiler_venta_venta.pickle')
    alquiler_model_path = os.path.join(PICKLE_DIR, 'random_forest_pipeline_alquiler_venta_alquiler.pickle')
    
    print(f"Contenido de {PICKLE_DIR}:")
    print(os.listdir(PICKLE_DIR))
    
    print(f"Buscando modelos en:")
    print(f"  - {venta_model_path}")
    print(f"  - {alquiler_model_path}")
    
    if not os.path.exists(venta_model_path):
        raise FileNotFoundError(f"El archivo de modelo de venta no se encuentra: {venta_model_path}")
    
    if not os.path.exists(alquiler_model_path):
        raise FileNotFoundError(f"El archivo de modelo de alquiler no se encuentra: {alquiler_model_path}")
    
    with open(venta_model_path, 'rb') as f:
        venta_model_dict = pickle.load(f)
    
    with open(alquiler_model_path, 'rb') as f:
        alquiler_model_dict = pickle.load(f)
    
    print("Estructura del modelo de venta:", type(venta_model_dict))
    print("Estructura del modelo de alquiler:", type(alquiler_model_dict))
    
    return venta_model_dict, alquiler_model_dict

def predict_with_model(model_dict, precio):
    if isinstance(model_dict, dict) and 'pipeline' in model_dict:
        pipeline = model_dict['pipeline']
        features = model_dict['features']
        
        # Crear un DataFrame con las características necesarias
        input_data = pd.DataFrame([[precio]], columns=['precio'])
        
        # Añadir columnas faltantes con valores por defecto
        for col, info in features['options_range'].items():
            if col not in input_data.columns:
                input_data[col] = info['default']
        
        # Asegurarse de que el DataFrame tiene todas las columnas necesarias en el orden correcto
        input_data = input_data[features['colums']]
        
        return pipeline.predict(input_data)[0]
    else:
        raise ValueError("El modelo no tiene la estructura esperada")

def update_predictions_and_ratios():
    engine = create_db_engine(DB_PRED)
    conn = engine.raw_connection()
    
    venta_model_dict, alquiler_model_dict = load_models()
    
    # Traer los datos de la tabla pred
    df_pred = pd.read_sql(f"SELECT * FROM {DB_PRED['TABLE']};", engine)

    for index, row in df_pred.iterrows():
        try:
            # Seleccionar el modelo adecuado en base al valor de alquiler_venta
            if row['alquiler_venta'] == 'venta':
                pred = predict_with_model(venta_model_dict, row['precio'])
            else:
                pred = predict_with_model(alquiler_model_dict, row['precio'])
            
            # Calcular el ratio
            ratio = pred / row['precio'] if row['precio'] != 0 else 0

            # Actualizar en la base de datos
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {DB_PRED['TABLE']}
                    SET predicion = %s, ratio = %s
                    WHERE id = %s;
                """, (pred, ratio, row['id']))
            conn.commit()
        except Exception as e:
            print(f"Error procesando fila {index}: {str(e)}")
            print("Datos de la fila:", row)
            continue
    
    conn.close()

def main():
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Proceso iniciado...")

    # Enviar mensaje de inicio a Telegram
    send_telegram_message("Iniciando proceso de predicciones")

    try:
        # Actualizar predicciones y ratios
        update_predictions_and_ratios()

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        completion_message = f"Proceso completado. Tiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        print(completion_message)
        send_telegram_message(completion_message)

    except Exception as e:
        error_message = f"Error en el proceso: {str(e)}"
        print(error_message)
        send_telegram_message(error_message)

if __name__ == '__main__':
    main()