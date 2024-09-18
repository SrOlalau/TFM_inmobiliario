import psycopg2
import pandas as pd
import pickle
import os
from sqlalchemy import create_engine, text

# Configuración de la base de datos de origen (datatuning)
DB_DEST = {
    "NAME": "datatuning",
    "USER": "datatuning",
    "PASSWORD": "datatuning",
    "HOST": "10.1.2.2",
    "PORT": "5444",
    "TABLE": "Datos_finales"
}

# Configuración de la base de datos de destino (pred)
DB_PRED = {
    "NAME": "pred",
    "USER": "pred",
    "PASSWORD": "pred",
    "HOST": "10.1.2.2",
    "PORT": "5445",
    "TABLE": "datos_finales_con_prediciones"
}

# Crear la URL de conexión para SQLAlchemy
def create_db_engine(db_config):
    engine_url = f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
    engine = create_engine(engine_url)
    return engine

# Comprobar si la tabla existe, si no, crearla
def check_and_create_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            );
        """)
        exists = cur.fetchone()[0]
        
        if not exists:
            cur.execute(f"""
                CREATE TABLE {table_name} AS 
                SELECT *, NULL::DOUBLE PRECISION AS predicion, NULL::DOUBLE PRECISION AS ratio 
                FROM "{DB_DEST['TABLE']}"  -- Comillas dobles para sensibilidad a mayúsculas
                LIMIT 0;
            """)
            conn.commit()
        return exists

# Cargar los datos de la base de datos de origen (datatuning)
def load_data_from_source():
    engine = create_db_engine(DB_DEST)
    query = f'SELECT * FROM "{DB_DEST["TABLE"]}";'  # Usar comillas dobles si es sensible a mayúsculas
    df = pd.read_sql(text(query), engine)
    engine.dispose()  # Cerrar la conexión
    return df

# Insertar los datos en la base de datos de destino (pred)
def insert_data_to_pred(df):
    engine = create_db_engine(DB_PRED)
    conn = engine.raw_connection()
    df.to_sql(DB_PRED['TABLE'], engine, if_exists='append', index=False)
    conn.commit()
    conn.close()

# Cargar modelos desde archivos pickle
def load_models(pickle_dir):
    with open(os.path.join(pickle_dir, 'random_forest_pipeline_alquiler_venta_venta.pkl'), 'rb') as f:
        venta_model = pickle.load(f)
    
    with open(os.path.join(pickle_dir, 'random_forest_pipeline_alquiler_venta_alquiler.pkl'), 'rb') as f:
        alquiler_model = pickle.load(f)
    
    return venta_model, alquiler_model

# Actualizar predicciones y ratio en la base de datos pred
def update_predictions_and_ratios(pickle_dir):
    engine = create_db_engine(DB_PRED)
    conn = engine.raw_connection()
    
    venta_model, alquiler_model = load_models(pickle_dir)
    
    # Traer los datos de la tabla pred
    df_pred = pd.read_sql(f"SELECT * FROM {DB_PRED['TABLE']};", engine)

    for index, row in df_pred.iterrows():
        # Seleccionar el modelo adecuado en base al valor de alquiler_venta
        if row['alquiler_venta'] == 'venta':
            pred = venta_model.predict([[row['precio']]])[0]
        else:
            pred = alquiler_model.predict([[row['precio']]])[0]
        
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
    
    conn.close()

# Función principal que ejecuta todo el flujo
def main(pickle_dir):
    # Paso 1: Conectarse a la base de datos 'pred' y comprobar/crear tabla
    engine_pred = create_db_engine(DB_PRED)
    conn_pred = engine_pred.raw_connection()
    check_and_create_table(conn_pred, DB_PRED["TABLE"])
    
    # Paso 2: Cargar los datos desde 'datatuning' y moverlos a 'pred'
    df_source = load_data_from_source()
    insert_data_to_pred(df_source)

    # Paso 3: Cargar los modelos y realizar las predicciones
    update_predictions_and_ratios(pickle_dir)

# Directorio donde están los pickles
pickle_dir = r"resultado"

# Ejecutar el proceso completo
main(pickle_dir)
