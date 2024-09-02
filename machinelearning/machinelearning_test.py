import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

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

def machine_learning(script_dir):
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df = pd.read_csv(file_path)
    print(f"Tamaño del DataFrame: {df.shape}")

    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df[~df['mt2'].isin([0, np.inf, -np.inf]) & df['mt2'].notna()]

    X = df.drop(['precio'], axis=1)
    y = df['precio']
    num_cols = X.select_dtypes(include=['number']).columns

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