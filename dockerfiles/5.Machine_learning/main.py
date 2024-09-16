import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

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
    """Preprocesa las características del DataFrame."""
    label_encoders = {}
    columns_to_drop = []
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.drop(target)
    datetime_columns = df.select_dtypes(include=['datetime64']).columns

    print("Procesando características...")
    for col in tqdm(categorical_columns, desc="Codificando variables categóricas"):
        df[col] = df[col].astype(str)
        unique_vals = df[col].nunique()
        if unique_vals > 20:
            columns_to_drop.append(col)
        else:
            df[col] = df[col].fillna('0')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            df[col] = df[col].astype(np.int8)  # Reducir tamaño de tipo de dato

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    df[numerical_columns] = df[numerical_columns].astype(np.float32)  # Reducir tamaño de tipo de dato

    for col in tqdm(datetime_columns, desc="Procesando columnas de fecha"):
        df[f'{col}_year'] = df[col].dt.year.astype(np.int16)
        df[f'{col}_month'] = df[col].dt.month.astype(np.int8)
        df[f'{col}_day'] = df[col].dt.day.astype(np.int8)
        columns_to_drop.append(col)

    for col in df.columns:
        if df[col].nunique() == 1:
            columns_to_drop.append(col)

    df = df.drop(columns=columns_to_drop)
    return df, scaler, label_encoders

def train_first_model(df, target):
    """Entrena un modelo inicial y selecciona las características más importantes."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando modelo inicial...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_50_features = feature_importances.nlargest(50).index

    # Liberar memoria eliminando el modelo inicial
    del model
    gc.collect()

    return top_50_features, X_train, X_test, y_train, y_test

def objective(trial, X, y):
    """Función objetivo para Optuna."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    score = -np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))
    del model
    gc.collect()
    return score

def optimize_hyperparameters(X, y):
    """Optimiza los hiperparámetros usando Optuna."""
    print("Optimizando hiperparámetros...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=5, show_progress_bar=True)
    
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params

def train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params):
    """Entrena el modelo con los mejores hiperparámetros y evalúa su rendimiento."""
    print("Entrenando modelo final...")
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluando modelo...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nEstadísticas del modelo:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 10 características más importantes:")
    print(feature_importances.head(10))
    
    print("\nAlgunas predicciones de RandomForest:")
    random_indices = np.random.choice(range(len(y_test)), size=6, replace=False)
    for idx in random_indices:
        real = y_test.iloc[idx]
        pred = y_pred[idx]
        print(f"Precio real: {real}, Precio predicho: {pred}")
    
    return model, feature_importances

def export_model(model, category, base_dir='/resultado'):
    """
    Exporta el modelo entrenado a un archivo pickle.
    
    Args:
    model: El modelo entrenado para exportar.
    category: La categoría del modelo (venta o alquiler).
    base_dir: El directorio base donde se guardará el modelo.
    
    Returns:
    str: La ruta completa donde se guardó el modelo.
    """
    directories = [
        base_dir,
        '/app/resultado',
        os.path.join(os.getcwd(), 'resultado'),
        os.path.expanduser('~/resultado')
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            model_filename = f'modelo_{category}.pickle'
            full_path = os.path.join(directory, model_filename)
            
            print(f"Guardando modelo en {full_path}...")
            with open(full_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Modelo guardado exitosamente en: {full_path}")
            return full_path
        except Exception as e:
            print(f"No se pudo guardar en {directory}. Error: {str(e)}")
    
    raise Exception("No se pudo guardar el modelo en ninguna ubicación.")

def main(target='precio', category='venta'):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Proceso iniciado...")

    # Enviar mensaje de inicio a Telegram
    send_telegram_message(f"Iniciando machine learning de {category}")

    df = load_data_from_postgres(category)
    
    print(f"\nEntrenando modelo para {category}...")

    df_processed, _, _ = process_features(df, target)
    del df  # Liberar memoria
    gc.collect()

    top_50_features, X_train, X_test, y_train, y_test = train_first_model(df_processed, target)
    del df_processed  # Liberar memoria
    gc.collect()

    X_train_top = X_train[top_50_features]
    X_test_top = X_test[top_50_features]
    del X_train, X_test  # Liberar memoria
    gc.collect()

    best_params = optimize_hyperparameters(X_train_top, y_train)

    final_model, feature_importances = train_and_evaluate_model(X_train_top, X_test_top, y_train, y_test, best_params)

    print("\nResumen del modelo:")
    print(f"Número de características utilizadas: {len(top_50_features)}")
    print(f"Tamaño del conjunto de entrenamiento: {X_train_top.shape}")
    print(f"Tamaño del conjunto de prueba: {X_test_top.shape}")
    print("\nMejores hiperparámetros:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    try:
        saved_path = export_model(final_model, category)
        print(f"Modelo guardado en: {saved_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {str(e)}")
        print("El modelo no se pudo guardar, pero el entrenamiento se completó.")

    # Construir mensaje de resumen para Telegram
    num_features_used = len(top_50_features)
    training_set_size = X_train_top.shape
    test_set_size = X_test_top.shape

    summary_message = "Machine learning finalizado\n\nResumen del modelo:\n"
    summary_message += f"Número de características utilizadas: {num_features_used}\n"
    summary_message += f"Tamaño del conjunto de entrenamiento: {training_set_size}\n"
    summary_message += f"Tamaño del conjunto de prueba: {test_set_size}\n"
    summary_message += "\nMejores hiperparámetros:\n"
    for param, value in best_params.items():
        summary_message += f"{param}: {value}\n"

    # Enviar mensaje de finalización a Telegram
    send_telegram_message(summary_message)

    # Liberar memoria final
    del X_train_top, X_test_top, y_train, y_test, final_model, feature_importances
    gc.collect()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == '__main__':
    # Cambia 'venta' por 'alquiler' si quieres entrenar el modelo para alquiler
    main(category='venta')
