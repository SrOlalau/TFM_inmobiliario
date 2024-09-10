import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
import warnings
import time
from datetime import datetime
from sqlalchemy import create_engine, text



warnings.filterwarnings('ignore')

# Configuración de la base de datos
DB_DEST = {
    "NAME": "datatuning",
    "USER": "datatuning",
    "PASSWORD": "datatuning",
    "HOST": "10.1.2.2",
    "PORT": "5444",
    "TABLE": "Datos_finales"
}

def load_data_from_postgres():
    """Carga los datos desde PostgreSQL."""
    connection_string = f"postgresql://{DB_DEST['USER']}:{DB_DEST['PASSWORD']}@{DB_DEST['HOST']}:{DB_DEST['PORT']}/{DB_DEST['NAME']}"
    engine = create_engine(connection_string)
    
    # Identificar columnas de fecha
    with engine.connect() as connection:
        query = text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{DB_DEST['TABLE']}' AND data_type LIKE '%timestamp%'")
        result = connection.execute(query)
        date_columns = [row[0] for row in result]

    # Cargar datos con parse_dates
    query = f'SELECT * FROM "{DB_DEST["TABLE"]}"'
    df = pd.read_sql(query, engine, parse_dates=date_columns)
    print(f"Tamaño del DataFrame cargado: {df.shape}")
    return df

def divide_dataset_bycategory(df, dummies):
    """Divide el dataset por categorías."""
    dummy_dfs = []
    if dummies:
        for dummy_col in dummies:
            unique_values = df[dummy_col].unique()
            for val in unique_values:
                df_subset = df[df[dummy_col] == val].copy()
                dummy_dfs.append((dummy_col, val, df_subset))
    return dummy_dfs if dummy_dfs else [(None, None, df)]

def process_features(df, target):
    """Preprocesa las características del DataFrame."""
    label_encoders = {}
    columns_to_drop = []
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.drop(target)
    datetime_columns = df.select_dtypes(include=['datetime64']).columns

    # Procesar columnas categóricas
    for col in categorical_columns:
        df[col] = df[col].astype(str)
        unique_vals = df[col].nunique()
        if unique_vals > 20:
            columns_to_drop.append(col)
        else:
            df[col] = df[col].fillna('0')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Procesar columnas numéricas
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Procesar columnas de fecha
    for col in datetime_columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        columns_to_drop.append(col)

    # Eliminar columnas con un solo valor
    for col in df.columns:
        if df[col].nunique() == 1:
            columns_to_drop.append(col)

    df = df.drop(columns=columns_to_drop)
    return df, scaler, label_encoders

def create_preprocessing_pipeline(df, target):
    """Crea un pipeline de preprocesamiento para las variables independientes."""
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
        ])

    return preprocessor

def train_first_model(df, target):
    """Entrena un modelo inicial y selecciona las características más importantes."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_50_features = feature_importances.nlargest(50).index

    return model, top_50_features, X_train, X_test, y_train, y_test

def objective(trial, X, y):
    """Función objetivo para Optuna."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    return -np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))

def optimize_hyperparameters(X, y):
    """Optimiza los hiperparámetros usando Optuna."""
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
    
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params

def train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params):
    """Entrena el modelo con los mejores hiperparámetros y evalúa su rendimiento."""
    model = RandomForestRegressor(** best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nEstadísticas del modelo:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 10 características más importantes:")
    print(feature_importances.head(10))

    print("\nAlgunas predicciones de RandomForest:")
    random_indices = np.random.choice(range(len(y_test)), size=6, replace=False)
    for idx in random_indices:
        real = y_test.iloc[idx]
        pred = y_pred[idx]
        print(f"Precio real: {real}, Precio predicho: {pred}")

    return model, feature_importances

def main(target='precio', dummies=['alquiler_venta']):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Proceso iniciado...")

    # Cargar datos
    df = load_data_from_postgres()
    
    # Dividir el dataset por categorías
    dummy_dfs = divide_dataset_bycategory(df, dummies)

    for dummy_info, val, df_dummy in dummy_dfs:
        print(f"\nEntrenando modelos para {dummy_info}={val}..." if dummy_info else "\nEntrenando modelo general...")

        # Preprocesar características
        df_processed, _, _ = process_features(df_dummy, target)

        # Entrenar el primer modelo y seleccionar las mejores características
        _, top_50_features, X_train, X_test, y_train, y_test = train_first_model(df_processed, target)

        # Usar solo las 50 mejores características
        X_train_top = X_train[top_50_features]
        X_test_top = X_test[top_50_features]

        # Optimizar hiperparámetros
        best_params = optimize_hyperparameters(X_train_top, y_train)

        # Entrenar y evaluar el modelo final
        final_model, feature_importances = train_and_evaluate_model(X_train_top, X_test_top, y_train, y_test, best_params)

        print("\nResumen del modelo:")
        print(f"Número de características utilizadas: {len(top_50_features)}")
        print(f"Tamaño del conjunto de entrenamiento: {X_train_top.shape}")
        print(f"Tamaño del conjunto de prueba: {X_test_top.shape}")
        print("\nMejores hiperparámetros:")
        for param, value in best_params.items():
            print(f"{param}: {value}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == '__main__':
    main()

