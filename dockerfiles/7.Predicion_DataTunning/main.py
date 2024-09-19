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

# Directorio donde están los pickles
PICKLE_DIR = "/resultados"

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
            # Obtener la estructura de la tabla de origen
            engine_source = create_db_engine(DB_DEST)
            with engine_source.connect() as conn_source:
                result = conn_source.execute(text(f"SELECT * FROM \"{DB_DEST['TABLE']}\" LIMIT 0"))
                column_names = result.keys()
            
            # Crear la tabla en la base de datos de destino
            columns = ', '.join([f"\"{col}\" VARCHAR" for col in column_names])
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    {columns},
                    predicion DOUBLE PRECISION,
                    ratio DOUBLE PRECISION
                );
            """)
            conn.commit()
        return exists

# Cargar los datos de la base de datos de origen (datatuning)
def load_data_from_source():
    engine = create_db_engine(DB_DEST)
    query = f'SELECT * FROM "{DB_DEST["TABLE"]}";'
    df = pd.read_sql(text(query), engine)
    engine.dispose()
    return df

# Insertar los datos en la base de datos de destino (pred)
def insert_data_to_pred(df):
    engine = create_db_engine(DB_PRED)
    conn = engine.raw_connection()
    df.to_sql(DB_PRED['TABLE'], engine, if_exists='append', index=False)
    conn.commit()
    conn.close()

# Cargar modelos desde archivos pickle
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
    
    # Cargar los diccionarios guardados en lugar de los modelos directamente
    with open(venta_model_path, 'rb') as f:
        venta_model_dict = pickle.load(f)
    
    with open(alquiler_model_path, 'rb') as f:
        alquiler_model_dict = pickle.load(f)
    
    # Extraer los pipelines de los diccionarios
    venta_model = venta_model_dict['pipeline']
    alquiler_model = alquiler_model_dict['pipeline']
    
    return venta_model, alquiler_model

# Actualizar predicciones y ratio en la base de datos pred
def update_predictions_and_ratios():
    engine = create_db_engine(DB_PRED)
    conn = engine.raw_connection()
    
    venta_model, alquiler_model = load_models()
    
    # Traer los datos de la tabla pred
    df_pred = pd.read_sql(f"SELECT * FROM {DB_PRED['TABLE']};", engine)

    for index, row in df_pred.iterrows():
        # Seleccionar el modelo adecuado en base al valor de alquiler_venta
        if row['alquiler_venta'] == 'venta':
            input_data = pd.DataFrame([row[['precio']]])  # Convertir la fila en DataFrame para el modelo
            pred = venta_model.predict(input_data)[0]
        else:
            input_data = pd.DataFrame([row[['precio']]])  # Convertir la fila en DataFrame para el modelo
            pred = alquiler_model.predict(input_data)[0]
        
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
def main():
    print("Iniciando el proceso principal...")
    
    # Paso 1: Conectarse a la base de datos 'pred' y comprobar/crear tabla
    print("Paso 1: Conectando a la base de datos 'pred' y comprobando/creando tabla...")
    engine_pred = create_db_engine(DB_PRED)
    conn_pred = engine_pred.raw_connection()
    table_existed = check_and_create_table(conn_pred, DB_PRED["TABLE"])
    print(f"La tabla {'ya existía' if table_existed else 'ha sido creada'}")
    
    # Paso 2: Cargar los datos desde 'datatuning' y moverlos a 'pred'
    print("Paso 2: Cargando datos desde 'datatuning' y moviéndolos a 'pred'...")
    df_source = load_data_from_source()
    print(f"Cargados {len(df_source)} registros de 'datatuning'")
    insert_data_to_pred(df_source)
    print("Datos insertados en 'pred'")

    # Paso 3: Cargar los modelos y realizar las predicciones
    print("Paso 3: Cargando modelos y realizando predicciones...")
    update_predictions_and_ratios()
    print("Predicciones actualizadas")

    print("Proceso completado con éxito")

# Ejecutar el proceso completo
if __name__ == "__main__":
    main()
