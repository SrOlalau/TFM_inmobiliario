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
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import numpy as np

def print_statistics(X_test,y_test, y_pred_rf,y_pred_lr, rf_pipeline):
    ## Evaluación con filtrado de variables
    # Filtrar filas donde 'latitude' o 'longitude' no sean NaN en el conjunto de prueba
    valid_indices = np.where(~X_test[['latitude', 'longitude']].isna().any(axis=1))[0]
    y_test_filtered = y_test.iloc[valid_indices]
    y_pred_rf_filtered = y_pred_rf[valid_indices]
    y_pred_lr_filtered = y_pred_lr[valid_indices]

    # Métricas para LinearRegression
    mae_lr_full = mean_absolute_error(y_test, y_pred_lr)
    mse_lr_full = mean_squared_error(y_test, y_pred_lr)
    rmse_lr_full = np.sqrt(mse_lr_full)
    r2_lr_full = r2_score(y_test, y_pred_lr)
    print("\nResultados de LinearRegression (sin filtrar):")
    print(f'MAE: {mae_lr_full}')
    print(f'RMSE: {rmse_lr_full}')
    print(f'R2 Score: {r2_lr_full}')

    # Métricas para LinearRegression usando solo las filas filtradas
    mae_lr_filtered = mean_absolute_error(y_test_filtered, y_pred_lr_filtered)
    mse_lr_filtered = mean_squared_error(y_test_filtered, y_pred_lr_filtered)
    rmse_lr_filtered = np.sqrt(mse_lr_filtered)
    r2_lr_filtered = r2_score(y_test_filtered, y_pred_lr_filtered)
    print("\nResultados de LinearRegression (filtrados lat y lon no son Nan):")
    print(f'MAE: {mae_lr_filtered}')
    print(f'RMSE: {rmse_lr_filtered}')
    print(f'R2 Score: {r2_lr_filtered}')

    ## Evaluación sin filtrar variables (resultados completos)
    # Métricas para RandomForest
    mae_rf_full = mean_absolute_error(y_test, y_pred_rf)
    mse_rf_full = mean_squared_error(y_test, y_pred_rf)
    rmse_rf_full = np.sqrt(mse_rf_full)
    r2_rf_full = r2_score(y_test, y_pred_rf)
    print("Resultados de RandomForest (sin filtrar):")
    print(f'MAE: {mae_rf_full}')
    print(f'RMSE: {rmse_rf_full}')
    print(f'R2 Score: {r2_rf_full}')

    # Métricas para RandomForest usando solo las filas filtradas
    mae_rf_filtered = mean_absolute_error(y_test_filtered, y_pred_rf_filtered)
    mse_rf_filtered = mean_squared_error(y_test_filtered, y_pred_rf_filtered)
    rmse_rf_filtered = np.sqrt(mse_rf_filtered)
    r2_rf_filtered = r2_score(y_test_filtered, y_pred_rf_filtered)
    print("\nResultados de RandomForest (filtrados lat y lon no son Nan):")
    print(f'MAE: {mae_rf_filtered}')
    print(f'RMSE: {rmse_rf_filtered}')
    print(f'R2 Score: {r2_rf_filtered}')

    # Imprimir algunas predicciones filtradas para verificar
    print("\nAlgunas predicciones de RandomForest (filtrados):")
    for i in range(min(5, len(valid_indices))):
        print(f"Precio real: {y_test_filtered.iloc[i]}, Precio predicho: {y_pred_rf_filtered[i]}")
    print("\nAlgunas predicciones de LinearRegression (filtrados):")
    for i in range(min(5, len(valid_indices))):
        print(f"Precio real: {y_test_filtered.iloc[i]}, Precio predicho: {y_pred_lr_filtered[i]}")

    # Obtener importancia de características de RandomForest
    rf_model = rf_pipeline.named_steps['regressor']
    importances = rf_model.feature_importances_
    feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Crear un DataFrame con la importancia de características
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("\nCaracterísticas más importantes de RandomForest:")
    print(importance_df.head(10))


def machine_learning(script_dir):
    # Ruta del archivo CSV
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data_DT.csv')

    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(file_path)
    # Verificar el tamaño del DataFrame
    print(f"Tamaño del DataFrame: {df.shape}")

    # Eliminar filas donde 'precio' es NaN, 0, o infinito
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df[~df['mt2'].isin([0, np.inf, -np.inf]) & df['mt2'].notna()]
    #df = df.drop(['planta', 'publicado_hace'], axis=1)
    
    # Separar características y variable objetivo
    X = df.drop(['precio'], axis=1)
    y = df['precio']

    # Identificar columnas numéricas
    num_cols = X.select_dtypes(include=['number']).columns

    # Preprocesamiento para columnas numéricas
    num_transformer = Pipeline(steps=[
    #    ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combinar preprocesadores solo para columnas numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('', num_transformer, num_cols)
        ])
    # Dividir el conjunto de datos en entrenamiento y prueba
    # Definir los hiperparámetros y sus rangos
    param_dist = {
        'regressor__n_estimators': [100, 150, 200, 500],
        'regressor__max_depth': [20, 30, 50, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__max_features': ['auto', 'sqrt', 'log2', 0.2, 0.5, 1.0],
        'regressor__bootstrap': [True, False],
        'regressor__criterion': ['squared_error']
    }

    # Crear el pipeline como antes
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Configurar el RandomizedSearchCV
    random_search = RandomizedSearchCV(
        rf_pipeline, 
        param_distributions=param_dist, 
        n_iter=50,  # Número de combinaciones aleatorias a probar
        cv=5,  # Número de folds para la validación cruzada
        verbose=3,  # Nivel de detalle de la salida
        n_jobs=-1,  # Usar todos los núcleos disponibles
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X[num_cols], y, test_size=0.2, random_state=42)
    # Entrenar con RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Mejor combinación de hiperparámetros
    print(f"Mejores hiperparámetros encontrados: {random_search.best_params_}")



def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    machine_learning(script_dir)

if __name__ == "__main__":
    main()