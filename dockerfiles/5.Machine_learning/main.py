import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text  # Nuevo: Para cargar datos desde PostgreSQL
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score  # cross_val_score para Optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import optuna  # Nuevo: Para optimización de hiperparámetros
import gc  # Nuevo: Manejo de memoria
import requests  # Nuevo: Para envío de mensajes a Telegram

warnings.filterwarnings('ignore')

mejor_cv = {
    'venta': {
        'bootstrap': False,
        'criterion': 'squared_error',
        'max_depth': 20,
        'max_features': 0.2,
        'min_samples_split': 2,
        'n_estimators': 150
    },
    'alquiler': {
        'bootstrap': False,
        'criterion': 'squared_error',
        'max_depth': 50,
        'max_features': 0.5,
        'min_samples_split': 2,
        'n_estimators': 150
    }
}

# Configuración de la base de datos (Nuevo)
DB_DEST = {
    "NAME": "datatuning",
    "USER": "datatuning",
    "PASSWORD": "datatuning",
    "HOST": "10.1.2.2",
    "PORT": "5444",
    "TABLE": "Datos_finales"
}

# Configuración de Telegram (Nuevo)
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

# Función de preprocesamiento general (Ya estaba)
def divide_dataset_bycategory(df, cat_cols_split_on=None):
    df_by_cat_cols = []
    if cat_cols_split_on:
        for col in cat_cols_split_on:
            unique_values = df[col].unique()
            for val in unique_values:
                df_subset = df[df[col] == val].copy()
                df_by_cat_cols.append((col, val, df_subset))
        return df_by_cat_cols
    else:
        return [(None, None, df)]

# Enviar mensaje a Telegram (Nuevo)
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


# Cargar datos desde PostgreSQL sin procesar las columnas de tipo fecha
def load_data_from_postgres():
    """Carga todos los datos desde PostgreSQL sin especificar columnas de fecha."""
    connection_string = f"postgresql://{DB_DEST['USER']}:{DB_DEST['PASSWORD']}@{DB_DEST['HOST']}:{DB_DEST['PORT']}/{DB_DEST['NAME']}"
    engine = create_engine(connection_string)

    # Cargar todos los datos de la tabla sin ningún filtro y sin procesar columnas de fecha
    query = text(f'SELECT * FROM "{DB_DEST["TABLE"]}"')
    df = pd.read_sql(query, engine)
    
    print(f"Tamaño del DataFrame cargado: {df.shape}")
    return df

# Crear pipeline para preprocesar variables (Ya estaba)
def create_preprocessing_pipeline(df, target):
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.drop(target)

    low_cardinality_cols = [col for col in categorical_columns if df[col].nunique() <= 20]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, low_cardinality_cols)
        ]
    )

    return preprocessor


# Crear pipeline para RandomForest (Ya estaba)
def create_pipeline(preprocessor, dummy_info):
    tipo_modelo = dummy_info[1]
    params_modelo = mejor_cv[tipo_modelo]

    model = RandomForestRegressor(**params_modelo)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline


# Entrenar el primer modelo para obtener las características más importantes (Ya estaba)
def train_first_model(df, target, dummy_info):
    preprocessor = create_preprocessing_pipeline(df, target)
    pipeline = create_pipeline(preprocessor, dummy_info=dummy_info)

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae, rmse, r2 = get_model_metrics(y_test, y_pred)

    feature_importances = pd.Series(pipeline.named_steps['regressor'].feature_importances_,
                                    index=pipeline.named_steps['preprocessor'].get_feature_names_out())
    top_50_features = feature_importances.nlargest(50).index

    return top_50_features


# Obtener métricas del modelo (Ya estaba)
def get_model_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


# Optimizador de hiperparámetros usando Optuna (Nuevo)
def objective(trial, X, y):
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

# Optimización de hiperparámetros con Optuna (Nuevo)
def optimize_hyperparameters(X, y):
    print("Optimizando hiperparámetros...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=1, show_progress_bar=True)
    
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params


# Entrenar el modelo final con los hiperparámetros de Optuna (Modificado)
def train_final_model(df, target, top_features, best_params, dummy_info=None):
    """
    Entrena el modelo final usando los hiperparámetros optimizados y guarda el pipeline en la ruta /resultado.
    
    Parámetros:
    - df: DataFrame original.
    - target: Nombre de la columna objetivo.
    - top_features: Lista de las características más importantes.
    - best_params: Hiperparámetros optimizados por Optuna.
    - dummy_info: Información sobre la categoría (opcional).
    """
    # Validar y coincidir las características importantes con las columnas originales
    matched_columns = validate_var_names(df, top_features)
    matched_columns.append(target)

    # Filtrar el DataFrame para incluir solo las columnas coincidentes
    df_filtered = df[matched_columns]

    # Crear preprocesador
    preprocessor = create_preprocessing_pipeline(df_filtered, target)

    # Aplicar los hiperparámetros optimizados por Optuna en lugar de los predeterminados
    model = RandomForestRegressor(**best_params, random_state=42)

    # Crear pipeline con el preprocesador y el modelo optimizado
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Dividir el conjunto de datos en entrenamiento y prueba
    X = df_filtered.drop(columns=[target])
    y = df_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = pipeline.predict(X_test)
    
    # Calcular métricas
    mae, rmse, r2 = get_model_metrics(y_test, y_pred)

    # Crear directorio 'resultado' si no existe
    models_dir = '/resultado'
    os.makedirs(models_dir, exist_ok=True)

    # Guardar el pipeline final en un archivo .pickle
    if dummy_info is None:
        pkl_filename = "random_forest_pipeline.pickle"
    else:
        pkl_filename = f"random_forest_pipeline_{dummy_info[0]}_{dummy_info[1]}.pickle"

    pickle_path = os.path.join(models_dir, pkl_filename)

    # Crear diccionario con el pipeline y las características
    features = {'colums': matched_columns,
                'options_range': range_describe(df, matched_columns)}

    to_save = {'pipeline': pipeline,
               'features': features}

    # Guardar el pipeline completo e información de las características en el archivo .pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)

    # Mostrar las métricas calculadas
    print_rf_stats(df_filtered, X_train, X_test, mae, rmse, r2, y_test, y_pred, pipeline)


# Función principal modificada para incluir la optimización de hiperparámetros con Optuna (Modificado)
def main(target='precio', dummies=['alquiler_venta']):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Proceso iniciado...")

    # Enviar mensaje de inicio a Telegram
    send_telegram_message(f"Iniciando machine learning de {dummies[0]}")

    # Cargar los datos desde la base de datos
    df = load_data_from_postgres(dummies[0])

    # Dividir el DataFrame por categoría
    dummy_dfs = divide_dataset_bycategory(df, dummies)

    for dummy_info, val, df_dummy in dummy_dfs:
        if dummy_info and val:
            print(f"Entrenando modelos para {dummy_info}={val}...")
        else:
            print("Entrenando modelo general...")

        # Entrenar el primer modelo y obtener las 50 características más importantes
        top_50_features = train_first_model(df_dummy, target, dummy_info=(dummy_info, val))

        # Optimizar los hiperparámetros usando Optuna
        X_train_top = df_dummy[top_50_features]
        y_train_top = df_dummy[target]
        best_params = optimize_hyperparameters(X_train_top, y_train_top)  # Modificado: Optimización de hiperparámetros

        # Entrenar el modelo final usando los hiperparámetros optimizados
        train_final_model(df_dummy, target, top_50_features, best_params, dummy_info=(dummy_info, val))  # Modificado: Pasar los hiperparámetros optimizados

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == '__main__':
    main()
