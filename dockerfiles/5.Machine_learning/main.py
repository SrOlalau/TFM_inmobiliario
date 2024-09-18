import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings
import time
from datetime import datetime
import pickle
import os
from tqdm import tqdm
import gc
import requests  # Importar requests para enviar mensajes a Telegram

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

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = 'TU_TELEGRAM_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'TU_TELEGRAM_CHAT_ID'

def send_telegram_message(message):
    """Envía un mensaje a un chat de Telegram."""
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

def load_data_from_postgres(category):
    """Carga los datos desde PostgreSQL filtrando por categoría."""
    connection_string = f"postgresql://{DB_DEST['USER']}:{DB_DEST['PASSWORD']}@{DB_DEST['HOST']}:{DB_DEST['PORT']}/{DB_DEST['NAME']}"
    engine = create_engine(connection_string)
    
    with engine.connect() as connection:
        query = text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{DB_DEST['TABLE']}' AND data_type LIKE '%timestamp%'")
        result = connection.execute(query)
        date_columns = [row[0] for row in result]

    # Envolver la consulta con text() para usar parámetros nombrados
    query = text(f'SELECT * FROM "{DB_DEST["TABLE"]}" WHERE alquiler_venta = :category')
    df = pd.read_sql(query, engine, params={'category': category}, parse_dates=date_columns)
    print(f"Tamaño del DataFrame cargado: {df.shape}")
    return df

def process_features(df, target):
    """Preprocesa las características del DataFrame y crea el preprocesador."""
    # Identificar columnas categóricas y numéricas
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
    
    # Eliminar columnas datetime
    if len(datetime_columns) > 0:
        print("Eliminando columnas datetime:", datetime_columns)
        df = df.drop(columns=datetime_columns)

    # Eliminar columnas no deseadas o con un único valor
    columns_to_drop = []
    for col in df.columns:
        if df[col].nunique() == 1 or df[col].isnull().all():
            columns_to_drop.append(col)
    if len(columns_to_drop) > 0:
        print("Eliminando columnas con un único valor o vacías:", columns_to_drop)
        df = df.drop(columns=columns_to_drop)

    # Actualizar listas de columnas después de eliminar
    categorical_columns = [col for col in categorical_columns if col in df.columns and col != target]
    numerical_columns = [col for col in numerical_columns if col in df.columns and col != target]

    # Manejar valores faltantes en variables categóricas
    df[categorical_columns] = df[categorical_columns].fillna('Desconocido')

    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
        ])

    # Lista de características finales
    features = numerical_columns + categorical_columns

    return df, preprocessor, features

def train_first_model(df, target, preprocessor):
    """Entrena un modelo inicial y selecciona las características más importantes."""
    X = df.drop(columns=[target])
    y = df[target]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Entrenar el modelo
    print("Entrenando el modelo inicial...")
    pipeline.fit(X_train, y_train)

    # Obtener los nombres de las características después del preprocesamiento
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Obtener importancias de características
    importances = pipeline.named_steps['regressor'].feature_importances_
    feature_importances = pd.Series(importances, index=feature_names)

    # Seleccionar las 50 características más importantes
    top_50_features = feature_importances.nlargest(50).index.tolist()

    return pipeline, top_50_features, X_train, X_test, y_train, y_test

def optimize_hyperparameters(pipeline, X, y):
    """Optimiza los hiperparámetros usando Optuna."""
    print("Optimizando hiperparámetros...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        pipeline.set_params(regressor__**params)
        scores = cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params

def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test, best_params):
    """Entrena el modelo con los mejores hiperparámetros y evalúa su rendimiento."""
    # Actualizar los hiperparámetros del modelo
    pipeline.set_params(regressor__**best_params)

    print("Entrenando el modelo final...")
    pipeline.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = pipeline.predict(X_test)

    # Evaluar el modelo
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nEstadísticas del modelo final:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    # Mostrar algunas predicciones
    print("\nAlgunas predicciones del modelo final:")
    for real, pred in zip(y_test[:5], y_pred[:5]):
        print(f"Precio real: {real}, Precio predicho: {pred}")

    # Obtener las importancias de características actualizadas
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = pipeline.named_steps['regressor'].feature_importances_
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("\nTop 10 características más importantes:")
    print(feature_importances.head(10))

    return pipeline, feature_importances

def export_model(model, features, category, base_dir='/resultado'):
    """
    Exporta el modelo entrenado y las características a un archivo pickle.
    """
    directories = [
        base_dir,
        '/app/resultado',
        os.path.join(os.getcwd(), 'resultado'),
        os.path.expanduser('~/resultado')
    ]

    model_data = {
        'pipeline': model,
        'features': features
    }

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            model_filename = f'modelo_{category}.pickle'
            full_path = os.path.join(directory, model_filename)

            print(f"Guardando modelo en {full_path}...")
            with open(full_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Modelo guardado exitosamente en: {full_path}")
            return full_path
        except Exception as e:
            print(f"No se pudo guardar en {directory}. Error: {str(e)}")

    raise Exception("No se pudo guardar el modelo en ninguna ubicación.")

def main(target='precio', category='alquiler'):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Proceso iniciado...")

    # Enviar mensaje de inicio a Telegram
    send_telegram_message(f"Iniciando entrenamiento de {category}")

    df = load_data_from_postgres(category)

    print(f"\nEntrenando modelo para {category}...")

    # Procesar características y crear el preprocesador
    df_processed, preprocessor, features = process_features(df, target)
    del df  # Liberar memoria
    gc.collect()

    # Entrenar el modelo inicial y seleccionar características importantes
    pipeline, top_features, X_train, X_test, y_train, y_test = train_first_model(df_processed, target, preprocessor)
    del df_processed  # Liberar memoria
    gc.collect()

    # Reducir X_train y X_test a las características seleccionadas
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Optimizar hiperparámetros
    best_params = optimize_hyperparameters(pipeline, X_train, y_train)

    # Entrenar y evaluar el modelo final
    final_pipeline, feature_importances = train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test, best_params)

    # Guardar el modelo
    try:
        saved_path = export_model(final_pipeline, top_features, category)
        print(f"Modelo guardado en: {saved_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {str(e)}")

    # Enviar mensaje de finalización a Telegram
    send_telegram_message(f"Entrenamiento completado para {category}")

    # Liberar memoria final
    del final_pipeline, X_train, X_test, y_train, y_test
    gc.collect()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == '__main__':
    main(category='alquiler')  # Cambia 'alquiler' por 'venta' según lo que necesites entrenar
