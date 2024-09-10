import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# Función de preprocesamiento general
def divide_dataset_bycategory(df, dummies): 
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

    # Procesar columnas de tipo 'object'
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

    # Procesar columnas numéricas
    numerical_columns = df.select_dtypes(include=[np.number]).columns
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_50_features = feature_importances.nlargest(50).index

    return model, top_50_features

def grid_search_cv_model(df, target, top_features, cv=5):
    X = df[top_features]
    y = df[target]

    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Definir los parámetros para la búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [100, 150, 200, 500],
        'max_depth': [20, 30, 50, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.5, 1.0],
        'bootstrap': [True, False],
        'criterion': ['squared_error']
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_dist, 
                               cv=cv, scoring='r2', n_jobs=-1, verbose=3)
    
    grid_search.fit(X, y)
    
    # Imprimir los mejores parámetros y el mejor score obtenido
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor R2 obtenido: {grid_search.best_score_}")
    
    return grid_search.best_estimator_

# Función principal para ejecutar el proceso completo
def main(target='precio', dummies=['alquiler_venta']):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(script_dir, '../datamunging/consolidated_data.csv')
    df = pd.read_csv(script_dir)
    print(f"Tamaño del DataFrame inicial: {df.shape}")
    
    dummy_dfs = divide_dataset_bycategory(df, dummies)

    for dummy_info, val, df_dummy in dummy_dfs:
        print(f"Entrenando modelos para {dummy_info}={val}..." if dummy_info else "Entrenando modelo general...")

        # Preprocesamiento de las características
        df_processed, scaler, label_encoders = process_features(df_dummy, target)

        # Entrenar el primer modelo
        _, top_50_features = train_first_model(df_processed, target)

        # Realizar la búsqueda de hiperparámetros con las 50 mejores características
        best_model = grid_search_cv_model(df_processed, target, top_50_features, cv=5)

if __name__ == '__main__':
    main()