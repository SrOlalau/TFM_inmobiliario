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
from warnings import simplefilter

pd.set_option('future.no_silent_downcasting', True)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

    # Extraer el top 50 de características más importantes
    top_50_features = importance_df.head(50)['feature'].tolist()
    
    return top_50_features

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
    # Paso 1: Encontrar la fecha más antigua en la columna 'fecha_extract'
    fecha_mas_antigua = df['fecha_extract'].min()

    # Paso 2: Calcular la diferencia en meses entre 'fecha_extract' y 'fecha_mas_antigua'
    df['mes_publicado'] = (df['fecha_extract'].dt.to_period('M') - fecha_mas_antigua.to_period('M')).apply(lambda x: x.n + 1)

    return df, label_encoders

def machine_learning(script_dir):
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Tamaño del DataFrame: {df.shape}")
    print("\nNúmero de columnas por tipo de dato:")
    print(df.dtypes.value_counts())
    
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df[~df['mt2'].isin([0, np.inf, -np.inf]) & df['mt2'].notna()]

    top_50_features_dict = {}

    for tipo in ['alquiler', 'venta']:
        print(f"\n--- Procesando para {tipo} ---\n")

        df_tipo = df[df['alquiler_venta'] == tipo].copy()
        
        # Aplicar el preprocesamiento al DataFrame
        df_tipo, label_encoders = preprocess_dataframe(df_tipo)

        X = df_tipo.drop(['precio'], axis=1)
        y = df_tipo['precio']

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
        print(f"Tamaño del DataFrame ({tipo}): {X.shape}")
        print(f"Tamaño de X_train ({tipo}): {X_train.shape}")
        print(f"Tamaño de X_test ({tipo}): {X_test.shape}")

        rf_pipeline.fit(X_train, y_train)
        y_pred_rf = rf_pipeline.predict(X_test)

        # Obtener el top 50 de los mejores predictores
        top_50_features = print_statistics(X_test, y_test, y_pred_rf, rf_pipeline)
        top_50_features_dict[tipo] = top_50_features

        # Verifica si el directorio 'models' existe y si no, créalo
        models_dir = os.path.join(script_dir, 'machinelearning/models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        model_file_path = os.path.join(models_dir, f'trained_model_{tipo}.pkl')
        with open(model_file_path, 'wb') as f:
            joblib.dump({
                'random_forest': rf_pipeline,
                'numeric_columns': num_cols,
                'top_50_features': top_50_features
            }, f, compress=3)  # El parámetro `compress` reduce el tamaño del archivo

        print(f"Modelos entrenados y guardados en 'trained_model_{tipo}.pkl'.")
        print('------------------------------------------------')
    
    return top_50_features_dict

def machine_learning_top_50(script_dir, top_50_features_dict):
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df = pd.read_csv(file_path , low_memory=False)
    
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df[~df['mt2'].isin([0, np.inf, -np.inf]) & df['mt2'].notna()]

    for tipo in ['alquiler', 'venta']:
        print(f"\n--- Procesando para {tipo} usando top 50 features ---\n")

        df_tipo = df[df['alquiler_venta'] == tipo].copy()
        
        # Aplicar el preprocesamiento al DataFrame
        df_tipo, label_encoders = preprocess_dataframe(df_tipo)

        # Seleccionar solo las columnas que están en el top 50 de características importantes
        top_50_features = top_50_features_dict[tipo]
        available_features = [col for col in top_50_features if col in df_tipo.columns]

        if not available_features:
            print(f"No hay características disponibles en el top 50 para {tipo} después del preprocesamiento.")
            continue  # Salta al siguiente tipo ('alquiler' o 'venta')

        df_tipo = df_tipo[available_features + ['precio']]

        X = df_tipo.drop(['precio'], axis=1)
        y = df_tipo['precio']

        num_cols = X.select_dtypes(include=['number']).columns  # Actualizar num_cols después del preprocesamiento

        if num_cols.empty:
            print(f"No hay columnas numéricas disponibles en X para {tipo}.")
            continue  # Salta al siguiente tipo ('alquiler' o 'venta')

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
        print(f"Tamaño del DataFrame ({tipo}): {X.shape}")
        print(f"Tamaño de X_train ({tipo}): {X_train.shape}")
        print(f"Tamaño de X_test ({tipo}): {X_test.shape}")

        rf_pipeline.fit(X_train, y_train)
        y_pred_rf = rf_pipeline.predict(X_test)

        print_statistics(X_test, y_test, y_pred_rf, rf_pipeline)

        # Verifica si el directorio 'models' existe y si no, créalo
        models_dir = os.path.join(script_dir, 'machinelearning/models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        model_file_path = os.path.join(models_dir, f'trained_model_top_50_{tipo}.pkl')
        with open(model_file_path, 'wb') as f:
            joblib.dump({
                'random_forest': rf_pipeline,
                'numeric_columns': num_cols,
                'top_50_features': top_50_features
            }, f, compress=3)  # El parámetro `compress` reduce el tamaño del archivo

        print(f"Modelos entrenados y guardados en 'trained_model_top_50_{tipo}.pkl'.")
        print('------------------------------------------------')

def main():
    # Incluir la fecha y hora en un print
    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Proceso iniciado...")
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Correr el primer machine_learning para obtener el top_50_features_dict
    top_50_features_dict = machine_learning(script_dir)
    
    # Ahora correr el machine_learning_top_50 basado en las top 50 características
    machine_learning_top_50(script_dir, top_50_features_dict)

if __name__ == "__main__":
    main()