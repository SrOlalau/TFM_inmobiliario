import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings('ignore')


# Función de preprocesamiento general
def divide_dataset_bycategory(df, cat_cols_split_on=None):
    #'dummies' es una lista de nombres de columnas en el dataframe a predecir
    # Divide el dataset entre Alquiler y Venta, devuelve una lista de Tuplas de 3 valores.
    # Cada Tupla tiene la siguiente estructura (nombre_columna, posibles_valores 0/1)
    # 1. Separar los datasets por las combinaciones de las variables dummy
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
            column_ranges[col] = {'range': [min_val, max_val], 'default': mean_val}

        else:
            # Si la columna no es categórica ni numérica, la omitimos
            continue

    return column_ranges


def get_model_metrics(y_test, y_pred):
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


def setup_models_folder():
    # Crear la carpeta 'models' si no existe
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


# Column transformer para preprocesar las variables independientes
def create_preprocessing_pipeline(df, target):
    # Obtener las columnas categóricas y numéricas
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.drop(target)

    # Filtrar columnas categóricas con menos o igual a 20 valores únicos
    low_cardinality_cols = [col for col in categorical_columns if df[col].nunique() <= 20]

    # Paso para las columnas categóricas: imputación y codificación de etiquetas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Paso para las columnas numéricas: imputación y escalado
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # ColumnTransformer: combinar los transformadores categóricos y numéricos
    # Nota, las columnas que no se especifican, no son parte del output (es un drop),
    # pasa con las columnas con más de 20 valores únicos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, low_cardinality_cols)
        ]
    )

    return preprocessor


# Junta preprocessor con modelo en un pipeline
def create_pipeline(preprocessor):
    # Definir el modelo de RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Crear el pipeline que combina el preprocesador y el modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline


# Función para entrenar el primer modelo
def train_first_model(df, target):
    # Crear el preprocesador
    preprocessor = create_preprocessing_pipeline(df, target)

    # Crear el pipeline completo
    pipeline = create_pipeline(preprocessor)

    # Dividir en conjunto de entrenamiento y prueba
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = pipeline.predict(X_test)

    # Calcular métricas
    mae, rmse, r2 = get_model_metrics(y_test, y_pred)

    # Obtener las características más importantes del modelo
    feature_importances = pd.Series(pipeline.named_steps['regressor'].feature_importances_,
                                    index=pipeline.named_steps['preprocessor'].get_feature_names_out())
    top_50_features = feature_importances.nlargest(50).index

    print("Top 50 características más importantes:")
    print(top_50_features)

    return top_50_features




# Función para entrenar el modelo final con las 50 mejores variables
def train_final_model(df, target, top_features, dummy_info=None):
    # Validate and match the top features with the original DataFrame columns
    matched_columns = validate_var_names(df, top_features)

    # Include the target column for training
    matched_columns.append(target)

    # Filter the DataFrame using the matched columns
    df_filtered = df[matched_columns]

    # Crear el preprocesador
    preprocessor = create_preprocessing_pipeline(df_filtered, target)

    # Crear el pipeline completo
    pipeline = create_pipeline(preprocessor)

    # Dividir en conjunto de entrenamiento y prueba
    X = df_filtered.drop(columns=[target])
    y = df_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el pipeline final
    pipeline.fit(X_train, y_train)

    # Crear carpeta de modelos si no existe
    models_dir = setup_models_folder()

    # Guardar el pipeline final en un archivo .pickle
    if dummy_info is None:
        pkl_filename = "random_forest_pipeline.pickle"
    else:
        pkl_filename = f"random_forest_pipeline_{dummy_info[0]}_{dummy_info[1]}.pickle"

    pickle_path = os.path.join(models_dir, pkl_filename)

    features = {'colums': matched_columns,
                'options_range': range_describe(df, matched_columns)}

    to_save = {'pipeline': pipeline,
               'features': features}

    # Guardar pipeline completo e información para describir features
    with open(pickle_path, 'wb') as f:
        pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)

    # Realizar predicciones
    y_pred = pipeline.predict(X_test)
    # Calcular métricas
    mae, rmse, r2 = get_model_metrics(y_test, y_pred)
    # Imprimir metricas
    print_rf_stats(df_filtered, X_train, X_test, mae, rmse, r2, y_test, y_pred, pipeline)


# Función principal para ejecutar el proceso completo
def main(target='precio', dummies=['alquiler_venta']):
    # Construir la ruta relativa al archivo CSV y cargar los datos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(script_dir, '../datamunging/consolidated_data.csv')

    df = pd.read_csv(script_dir)
    print(f"Tamaño del DataFrame inicial: {df.shape}")

    dummy_dfs = divide_dataset_bycategory(df, dummies)

    for dummy_info, val, df_dummy in dummy_dfs:
        print(f"Entrenando modelos para {dummy_info}={val}..." if dummy_info else "Entrenando modelo general...")

        # Entrenar el primer modelo y obtener las 50 características más importantes
        top_50_features = train_first_model(df_dummy, target)

        # Entrenar el modelo final usando las 50 mejores características
        train_final_model(df_dummy, target, top_50_features, dummy_info=(dummy_info, val))



    
if __name__ == '__main__':
    start_time = time.time()  # Captura el tiempo de inicio
    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Proceso iniciado...")

    main()

    end_time = time.time()  # Captura el tiempo de finalización
    elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido
    # Convertir el tiempo transcurrido a horas, minutos y segundos
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")