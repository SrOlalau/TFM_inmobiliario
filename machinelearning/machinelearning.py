import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import warnings
from datetime import datetime
import os
import time

warnings.filterwarnings('ignore')

# Función de preprocesamiento general
def divide_dataset_bycategory(df, dummies): 
    #'dummies' es una lista de nombres de columnas en el dataframe a predecir
    # Divide el dataset entre Alquiler y Venta, devuelve una lista de Tuplas de 3 valores.
    # Cada Tupla tiene la siguiente estructura (nombre_columna, posibles_valores 0/1)
    # 1. Separar los datasets por las combinaciones de las variables dummy
    dummy_dfs = []
    if dummies:
        for dummy_col in dummies:
            unique_values = df[dummy_col].unique()
            for val in unique_values:
                df_subset = df[df[dummy_col] == val].copy()
                dummy_dfs.append((dummy_col, val, df_subset))

    return dummy_dfs if dummy_dfs else [(None, None, df)]

# Función para preprocesar las variables independientes
def process_features(df, target):
    label_encoders = {}
    columns_to_drop = []
    categorical_columns = df.select_dtypes(include=['object']).columns

    # 3.1 Procesar columnas de tipo 'object'
    for col in categorical_columns:
        df[col] = df[col].astype(str)  # Asegúrate de que todos los valores sean cadenas de texto
        unique_vals = df[col].nunique()
        if unique_vals > 20:
            columns_to_drop.append(col)
        else:
            df[col] = df[col].fillna('0')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # 3.2 Procesar columnas numéricas
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    # Excluir la columna objetivo de las columnas a escalar
    numerical_columns = numerical_columns.drop(target)

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Eliminar columnas con un solo valor
    for col in df.columns:
        if df[col].nunique() == 1:
            columns_to_drop.append(col)

    df = df.drop(columns=columns_to_drop)
    return df, scaler, label_encoders

# Función para entrenar el primer modelo
def train_first_model(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#-----> NECESITAMOS HACER SOLO SOBRE X_TRAIN Por ende aqui se hace el preprocesado
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#-----> Combinar el preprocesamiento de los datos (standardscaler, label encoder, imputacion) con el modelo a entrenar   
    model.fit(X_train, y_train)
#-----> Pipeline.fit(X_train, y_train) Signigica que se estan entrenando los encoders, imputers y modelo
    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Extraer las 50 variables más importantes
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_50_features = feature_importances.nlargest(50).index

    return model, top_50_features, X_train, X_test, y_train, y_test, y_pred, mae, rmse, r2

# Función para entrenar el modelo final con las 50 mejores variables
def train_final_model(df, target, top_features, scaler, label_encoders, dummy_info=None):
    X = df[top_features]
    y = df[target]

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#-----> NECESITAMOS HACER SOLO SOBRE X_TRAIN Por ende aqui se hace el preprocesado
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Crear la carpeta 'models' si no existe
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Guardar el modelo en un archivo .pkl dentro de la carpeta 'models'
    pkl_filename = f"random_forest_model.pkl" if not dummy_info else f"random_forest_model_{dummy_info[0]}_{dummy_info[1]}.pkl"
    pkl_path = os.path.join(models_dir, pkl_filename)
    dump((model, scaler, label_encoders), pkl_path)

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
    for real, pred in zip(y_test.head(5), y_pred[:5]):
        print(f"Precio real: {real}, Precio predicho: {pred}")

    print("\nCaracterísticas más importantes de RandomForest:")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features_df = feature_importances.nlargest(50).reset_index()
    top_features_df.columns = ['feature', 'importance']
    print(top_features_df.head(10))
    print("------------------------------------------------")

# Función principal para ejecutar el proceso completo
def main(target='precio', dummies=['alquiler_venta']):
    # Construir la ruta relativa al archivo CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(script_dir, '../datamunging/consolidated_data.csv')  # Ajusta la ruta según la estructura de tu proyecto
    start_time = time.time()  # Captura el tiempo de inicio
    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Proceso iniciado...")
    df = pd.read_csv(script_dir)  # Cargar el DataFrame desde el archivo CSV
    print(f"Tamaño del DataFrame inicial: {df.shape}")
    
    dummy_dfs = divide_dataset_bycategory(df, dummies)

    for dummy_info, val, df_dummy in dummy_dfs:
        print(f"Entrenando modelos para {dummy_info}={val}..." if dummy_info else "Entrenando modelo general...")

        # Preprocesamiento de las características
        df_processed, scaler, label_encoders = process_features(df_dummy, target)

        # Entrenar el primer modelo
        model, top_50_features, X_train, X_test, y_train, y_test, y_pred, mae, rmse, r2 = train_first_model(df_processed, target)

        # Entrenar el modelo final usando las 50 mejores características
        train_final_model(df_processed, target, top_50_features, scaler, label_encoders, dummy_info=(dummy_info, val))

    end_time = time.time()  # Captura el tiempo de finalización
    elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido

    # Convertir el tiempo transcurrido a horas, minutos y segundos
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
if __name__ == '__main__':
    main()