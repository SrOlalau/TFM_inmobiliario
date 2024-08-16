import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import os

def machine_learning(script_dir):

    # Ruta del archivo CSV
    file_path =  os.path.join(script_dir, 'datatuning/datatuning.csv')

    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(file_path)

    # Verificar el tamaño del DataFrame
    print(f"Tamaño del DataFrame: {df.shape}")

    # Eliminar filas donde 'precio' es NaN, 0, o infinito
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]

    # Separar características y variable objetivo
    X = df.drop('precio', axis=1)
    y = df['precio']

    # Imputar valores NaN en y con la mediana
    y = y.fillna(y.median())

    # Identificar columnas numéricas
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Preprocesamiento para columnas numéricas
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    # Combinar preprocesadores solo para columnas numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols)
        ])

    # Crear pipelines para RandomForest y LinearRegression
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X[num_cols], y, test_size=0.2, random_state=42)

    # Verificar tamaños de conjuntos de entrenamiento y prueba
    print(f"Tamaño de X_train: {X_train.shape}")
    print(f"Tamaño de X_test: {X_test.shape}")

    # Entrenar y evaluar el modelo de RandomForest
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print("Resultados de RandomForest:")
    print(f'MAE: {mae_rf}')
    print(f'RMSE: {rmse_rf}')
    print(f'R2 Score: {r2_rf}')

    # Entrenar y evaluar el modelo de LinearRegression
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print("\nResultados de LinearRegression:")
    print(f'MAE: {mae_lr}')
    print(f'RMSE: {rmse_lr}')
    print(f'R2 Score: {r2_lr}')

    # Imprimir algunas predicciones para verificar
    print("\nAlgunas predicciones de RandomForest:")
    for i in range(5):
        print(f"Precio real: {y_test.iloc[i]}, Precio predicho: {y_pred_rf[i]}")

    print("\nAlgunas predicciones de LinearRegression:")
    for i in range(5):
        print(f"Precio real: {y_test.iloc[i]}, Precio predicho: {y_pred_lr[i]}")

    # Obtener importancia de características de RandomForest
    rf_model = rf_pipeline.named_steps['regressor']
    importances = rf_model.feature_importances_
    feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Crear un DataFrame con la importancia de características
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)

    print("\nCaracterísticas más importantes de RandomForest:")
    print(importance_df.head(10))

    print("Modelos completados.")

def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    machine_learning(script_dir)


if __name__ == "__main__":
    main()