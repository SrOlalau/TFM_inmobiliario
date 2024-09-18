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
#Original
def print_rf_stats(df, X_train, X_test, mae, rmse, r2, y_test, y_pred, pipeline):
    # Imprimir resultados
    print("-----------DataFrame-----------:")
    print(f"Tamaño del DataFrame: {df.shape}")
    print(f"Tamaño de X_train: {X_train.shape}")
    print(f"Tamaño de X_test: {X_test.shape}")
    print("-----------Resultados de RandomForest:-----------")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    print("\nAlgunas predicciones de RandomForest:")
    # Generar 5 índices aleatorios únicos de y_test
    random_indices = np.random.choice(range(len(y_test)), size=6, replace=False)

    # Mostrar las predicciones correspondientes a los índices aleatorios
    for idx in random_indices:
        real = y_test.iloc[idx]  # Acceso directo para pandas Series
        pred = y_pred[idx]  # Acceso directo para numpy array
        print(f"Precio real: {real}, Precio predicho: {pred}")

    print("\nCaracterísticas más importantes de RandomForest:")
    model_cols = []
    for col in pipeline.named_steps['preprocessor'].get_feature_names_out():
        # Remove prefixes "cat__" or "num__"
        stripped_col = col.split('__', 1)[-1]
        model_cols.append(stripped_col)
    feature_importances = pd.Series(pipeline.named_steps['regressor'].feature_importances_,
                                    index= model_cols)
    top_features_df = feature_importances.nlargest(50).reset_index()
    top_features_df.columns = ['feature', 'importance']
    print(top_features_df.head(10))
    print("------------------------------------------------")


#Original
def range_describe(df, matched_columns):
    """
    Crea un diccionario con el rango de valores, así como la media o moda para cada columna del DataFrame.

    Parameters:
    df (DataFrame): El DataFrame original.
    matched_columns (list): Lista de columnas del DataFrame que se están utilizando en el modelo.

    Returns:
    dict: Diccionario con el nombre de cada columna, su rango de valores/opciones posibles, y la media o moda.
    """
    column_ranges = {}

    for col in matched_columns:
        if df[col].dtype == 'object':  # Columnas categóricas
            # Obtener valores únicos y convertirlos a lista para el diccionario
            unique_values = df[col].unique().tolist()
            # Calcular la moda
            mode_val = df[col].mode()[0] if not df[col].mode().empty else unique_values[0]
            column_ranges[col] = {'range': unique_values, 'default': mode_val}

        elif pd.api.types.is_numeric_dtype(df[col]):  # Columnas numéricas
            # Calcular mínimo, máximo y media
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            perc_1 = df[col].quantile(0.01)
            perc_95 = df[col].quantile(0.95)

            column_ranges[col] = {
                'range': [min_val, max_val],
                'default': mean_val,
                'range_80pct': [perc_1, perc_95]  # Rango del 10 al 90 percentil
            }

        else:
            # Si la columna no es categórica ni numérica, la omitimos
            continue

    return column_ranges
#Original
def validate_var_names(df, model_columns):
    """
    Validate and match the columns from the original DataFrame (df) with the columns used in the model.

    Parameters:
    df (DataFrame): Original DataFrame before preprocessing.
    model_columns (Index or list): Columns used in the model after preprocessing.

    Returns:
    list: List of column names from the original DataFrame that correspond to the model columns.
    """
    # Initialize lists for matching columns
    matched_columns = []

    # Iterate through all model columns
    for col in model_columns:
        # Remove prefixes "cat__" or "num__"
        stripped_col = col.split('__', 1)[-1]

        # Check for numerical columns directly
        if col.startswith('num__') and stripped_col in df.columns:
            matched_columns.append(stripped_col)

        # Check for categorical columns using "startswith"
        elif col.startswith('cat__'):
            # Find any column in df that starts with the stripped column name
            for df_col in df.columns:
                if stripped_col.startswith(df_col):
                    matched_columns.append(df_col)

    # Remove duplicates if any
    matched_columns = list(set(matched_columns))
    return matched_columns

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

    model = RandomForestRegressor(**params_modelo,n_jobs=-1)

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


# Optimizador de hiperparámetros usando Optuna (Modificado)
def objective(trial, X, y, preprocessor):
    """
    Función objetivo para Optuna, que incluye el preprocesador para transformar los datos.
    
    Parámetros:
    - trial: Parámetros del trial de Optuna.
    - X: Variables independientes.
    - y: Variable dependiente.
    - preprocessor: El preprocesador (pipeline) que se aplicará a los datos.
    
    Devuelve:
    - score: El error medio cuadrático negativo promedio del modelo.
    """
    # Parámetros sugeridos por Optuna para RandomForest
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),  # 100,1000
        'max_depth': trial.suggest_int('max_depth', 10, 20),          #10,100
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'n_jobs': -1  # Usar todos los núcleos disponibles
    }

    # Crear el modelo de RandomForest con los hiperparámetros optimizados
    model = RandomForestRegressor(**params, random_state=42)

    # Crear el pipeline con el preprocesador y el modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Agregamos el preprocesador al pipeline
        ('regressor', model)
    ])

    # Evaluar el modelo con validación cruzada
    score = -np.mean(cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))

    # Limpiar el modelo para liberar memoria
    del model
    gc.collect()

    return score


# Optimización de hiperparámetros con Optuna (Modificado)
def optimize_hyperparameters(X, y, preprocessor):
    """
    Optimiza los hiperparámetros usando Optuna, pasando el preprocesador.
    
    Parámetros:
    - X: Variables independientes.
    - y: Variable dependiente.
    - preprocessor: Preprocesador (pipeline) a aplicar a los datos.
    
    Devuelve:
    - Los mejores hiperparámetros encontrados.
    """
    print("Optimizando hiperparámetros...")
    
    # Crear un estudio de Optuna
    study = optuna.create_study(direction='minimize')

    # Optimizar los hiperparámetros con el preprocesador
    study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=1, show_progress_bar=True)

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

    # Cargar los datos desde la base de datos y limpiar las columnas numéricas
    df = load_data_from_postgres()
    

    # Dividir el DataFrame por categoría
    dummy_dfs = divide_dataset_bycategory(df, dummies)

    for dummy_info, val, df_dummy in dummy_dfs:
        if dummy_info and val:
            print(f"Entrenando modelos para {dummy_info}={val}...")
        else:
            print("Entrenando modelo general...")

        # Entrenar el primer modelo y obtener las 50 características más importantes
        top_50_features = train_first_model(df_dummy, target, dummy_info=(dummy_info, val))
        matched_columns=validate_var_names(df_dummy, top_50_features)
        # Crear el preprocesador para este conjunto de datos
        preprocessor = create_preprocessing_pipeline(df_dummy[[*matched_columns,target]], target)
        # Optimizar los hiperparámetros usando Optuna
        X_train_top = df_dummy[matched_columns]
        y_train_top = df_dummy[target]
        best_params = optimize_hyperparameters(X_train_top, y_train_top, preprocessor)  # Modificado: Pasa el preprocesador

        # Entrenar el modelo final usando los hiperparámetros optimizados
        train_final_model(df_dummy, target, top_50_features, best_params, dummy_info=(dummy_info, val))  # Modificado: Pasar los hiperparámetros optimizados

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == '__main__':
    main()
