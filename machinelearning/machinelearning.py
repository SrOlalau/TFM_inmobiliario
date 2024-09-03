import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

def print_statistics(X_test, y_test, y_pred_rf, rf_pipeline):
    valid_indices = np.where(~X_test[['latitude', 'longitude']].isna().any(axis=1))[0]
    y_test_filtered = y_test.iloc[valid_indices]
    y_pred_rf_filtered = y_pred_rf[valid_indices]

    mae_rf_full = mean_absolute_error(y_test, y_pred_rf)
    mse_rf_full = mean_squared_error(y_test, y_pred_rf)
    rmse_rf_full = np.sqrt(mse_rf_full)
    r2_rf_full = r2_score(y_test, y_pred_rf)
    print("Resultados de RandomForest (sin filtrar):")
    print(f'MAE: {mae_rf_full}')
    print(f'RMSE: {rmse_rf_full}')
    print(f'R2 Score: {r2_rf_full}')

    mae_rf_filtered = mean_absolute_error(y_test_filtered, y_pred_rf_filtered)
    mse_rf_filtered = mean_squared_error(y_test_filtered, y_pred_rf_filtered)
    rmse_rf_filtered = np.sqrt(mse_rf_filtered)
    r2_rf_filtered = r2_score(y_test_filtered, y_pred_rf_filtered)
    print("\nResultados de RandomForest (filtrados lat y lon no son Nan):")
    print(f'MAE: {mae_rf_filtered}')
    print(f'RMSE: {rmse_rf_filtered}')
    print(f'R2 Score: {r2_rf_filtered}')

    print("\nAlgunas predicciones de RandomForest (filtrados):")
    for i in range(min(5, len(valid_indices))):
        print(f"Precio real: {y_test_filtered.iloc[i]}, Precio predicho: {y_pred_rf_filtered[i]}")

    rf_model = rf_pipeline.named_steps['regressor']
    importances = rf_model.feature_importances_
    feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("\nCaracterísticas más importantes de RandomForest:")
    print(importance_df.head(10))

def preprocess_dataframe(df):
    # 1. Transformaciones en columnas object

    # 1.1 Convertir 'fecha_extract' al formato de fecha (datetime)
    df['fecha_extract'] = pd.to_datetime(df['fecha_extract'], format='%Y-%m-%d')

    # 1.2 Filtrar y eliminar columnas object que tengan más de 20 valores únicos
    columns_to_drop = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 20]
    df.drop(columns=columns_to_drop, inplace=True)

    # 1.3 Eliminar columnas que tengan un único valor
    columns_with_single_value = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=columns_with_single_value, inplace=True)

    # 1.4 Convertir columnas con 2 valores únicos a booleanas
    for col in df.columns:
        if df[col].nunique() == 2:
            unique_values = df[col].unique()
            df[col] = df[col] == unique_values[0]  # Convertir al primer valor como True, el segundo como False

    # 1.5 Reemplazar valores NaN con 0 en columnas de tipo object
    df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).fillna(0)

    # 1.6 Transformar columnas object en valores numéricos usando LabelEncoder
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 2. Transformaciones en columnas float

    # 2.1 Crear indicadores de NaN para cada columna numérica
    for col in df.select_dtypes(include=['float']).columns:
        df[f'{col}_is_nan'] = df[col].isna().astype(int)

    # 2.2 Crear la columna 'dias_extraccion' que es la diferencia en días desde la fecha actual
    df['dias_extraccion'] = (datetime.now() - df['fecha_extract']).dt.days

    return df, label_encoders

def machine_learning(script_dir):
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df = pd.read_csv(file_path)
    print(f"Tamaño del DataFrame: {df.shape}")

    print("Tipos de datos de cada columna:")
    print(df.dtypes)

    print("\nNúmero de columnas por tipo de dato:")
    print(df.dtypes.value_counts())
    
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df[~df['mt2'].isin([0, np.inf, -np.inf]) & df['mt2'].notna()]

    # Aplicar el preprocesamiento al DataFrame
    df, label_encoders = preprocess_dataframe(df)
    
    X = df.drop(['precio'], axis=1)
    y = df['precio']

    num_cols = X.select_dtypes(include=['number']).columns  # Actualizar num_cols después del preprocesamiento

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols)
        ])

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(bootstrap=True, max_depth=None, max_features=0.5,
                                            min_samples_split=2, n_estimators=150, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X[num_cols], y, test_size=0.2, random_state=42)
    print(f"Tamaño del DataFrame: {X.shape}")
    print(f"Tamaño de X_train: {X_train.shape}")
    print(f"Tamaño de X_test: {X_test.shape}")

    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    print_statistics(X_test, y_test, y_pred_rf, rf_pipeline)

    # Verifica si el directorio 'models' existe y si no, créalo
    models_dir = os.path.join(script_dir, 'machinelearning/models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_file_path = os.path.join(models_dir, 'trained_models.pkl')
    with open(model_file_path, 'wb') as f:
        joblib.dump({
            'random_forest': rf_pipeline,
            'numeric_columns': num_cols
        }, f, compress=3)  # El parámetro `compress` reduce el tamaño del archivo

    print("Modelos entrenados y guardados en 'trained_models.pkl'.")

def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    machine_learning(script_dir)

if __name__ == "__main__":
    main()