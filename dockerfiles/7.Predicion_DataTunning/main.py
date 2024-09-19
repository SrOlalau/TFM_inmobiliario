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

# Cargar todos los datos de la base de datos de origen (datatuning)
def load_data_from_source():
    engine = create_db_engine(DB_DEST)
    query = f'SELECT * FROM "{DB_DEST["TABLE"]}";'
    
    # Cargar todos los datos de la tabla en RAM
    df = pd.read_sql(text(query), engine)
    
    engine.dispose()
    return df

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
    
    # Cargar los modelos
    venta_model, alquiler_model = load_models()
    
    # Cargar toda la base de datos en RAM
    df_pred = load_data_from_source()  # Ahora carga toda la base de datos

    # Filtrar datos según el tipo de operación: alquiler o venta
    df_pred_alquiler = df_pred[df_pred['alquiler_venta'] == 'alquiler'].copy()  # Copia para evitar errores
    df_pred_ventas = df_pred[df_pred['alquiler_venta'] == 'venta'].copy()       # Copia para evitar errores

    # Generar predicciones para alquiler y ventas
    if not df_pred_alquiler.empty:
        df_pred_alquiler['predicion'] = alquiler_model.predict(df_pred_alquiler)
        df_pred_alquiler['ratio'] = df_pred_alquiler['predicion'] / df_pred_alquiler['precio']

    if not df_pred_ventas.empty:
        df_pred_ventas['predicion'] = venta_model.predict(df_pred_ventas)
        df_pred_ventas['ratio'] = df_pred_ventas['predicion'] / df_pred_ventas['precio']
    
    # Insertar o actualizar las predicciones en la tabla pred, manteniendo la estructura original
    upsert_predictions_to_db(df_pred_alquiler, df_pred_ventas, engine)

def upsert_predictions_to_db(df_pred_alquiler, df_pred_ventas, engine):
    conn = engine.connect()

    # Limpiar la tabla destino antes de la inserción
    conn.execute(text(f"DELETE FROM {DB_PRED['TABLE']}"))

    # Si hay datos de alquiler, hacer la inserción directa en la tabla de predicciones
    if not df_pred_alquiler.empty:
        df_pred_alquiler.to_sql(DB_PRED['TABLE'], conn, if_exists='append', index=False)

    # Si hay datos de ventas, hacer la inserción directa en la tabla de predicciones
    if not df_pred_ventas.empty:
        df_pred_ventas.to_sql(DB_PRED['TABLE'], conn, if_exists='append', index=False)

    conn.close()


# Cargar los datos de la base de datos en RAM y tomar una muestra de 10 filas
def load_data_sample():
    engine = create_db_engine(DB_DEST)
    query = f'SELECT * FROM "{DB_DEST["TABLE"]}";'
    
    # Cargar todos los datos en RAM
    df = pd.read_sql(text(query), engine)
    
    # Tomar una muestra aleatoria de 10 filas para pruebas
    df_sample = df.sample(n=10, random_state=42)
    
    engine.dispose()
    return df_sample


# Función principal que ejecuta todo el flujo
def main():
    print("Iniciando el proceso principal...")
    
    # Paso 1: Conectarse a la base de datos 'pred' y comprobar/crear tabla
    print("Paso 1: Conectando a la base de datos 'pred' y comprobando/creando tabla...")
    engine_pred = create_db_engine(DB_PRED)
    conn_pred = engine_pred.raw_connection()
    table_existed = check_and_create_table(conn_pred, DB_PRED["TABLE"])
    print(f"La tabla {'ya existía' if table_existed else 'ha sido creada'}")
    
    # Paso 2: Cargar los modelos y realizar las predicciones
    print("Paso 2: Cargando modelos y realizando predicciones...")
    update_predictions_and_ratios()
    print("Predicciones actualizadas")

    print("Proceso completado con éxito")

# Ejecutar el proceso completo
if __name__ == "__main__":
    main()
