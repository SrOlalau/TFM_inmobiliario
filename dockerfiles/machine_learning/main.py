import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
from optuna.integration import OptunaSearchCV
import warnings
import time
from datetime import datetime
import pickle
import os

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
    
    with engine.connect() as connection:
        query = text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{DB_DEST['TABLE']}' AND data_type LIKE '%timestamp%'")
        result = connection.execute(query)
        date_columns = [row[0] for row in result]

    query = f'SELECT * FROM "{DB_DEST["TABLE"]}"'
    df = pd.read_sql(query, engine, parse_dates=date_columns)
    print(f"Tamaño del DataFrame cargado: {df.shape}")
    return df

def divide_dataset_bycategory(df, category):
    """Divide el dataset por categorías."""
    df_subset = df[df['alquiler_venta'] == category].copy()
    return df_subset

def process_features(df, target):
    """Preprocesa las características del DataFrame."""
    label_encoders = {}
    columns_to_drop = []
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.drop(target)
    datetime_columns = df.select_dtypes(include=['datetime64']).columns

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

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    for col in datetime_columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        columns_to_drop.append(col)

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

def optimize_hyperparameters(X, y):
    """Optimiza los hiperparámetros usando OptunaSearchCV."""
    param_distributions = {
        'n_estimators': optuna.distributions.IntDistribution(100, 1000),
        'max_depth': optuna.distributions.IntDistribution(10, 100),
        'min_samples_split': optuna.distributions.IntDistribution(2, 10),
        'min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
        'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2', None]),
        'bootstrap': optuna.distributions.CategoricalDistribution([True, False])
    }

    optuna_search = OptunaSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions,
        n_trials=75,
        cv=5,
        n_jobs=-1,
        random_state=42
    )

    optuna_search.fit(X, y)
    
    print("Mejores hiperparámetros encontrados:")
    print(optuna_search.best_params_)
    return optuna_search.best_estimator_

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    """Evalúa el rendimiento del modelo."""
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

def main(target='precio', category='venta'):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Proceso iniciado...")

    df = load_data_from_postgres()
    
    df_category = divide_dataset_bycategory(df, category)

    print(f"\nEntrenando modelo para {category}...")

    df_processed, _, _ = process_features(df_category, target)

    _, top_50_features, X_train, X_test, y_train, y_test = train_first_model(df_processed, target)

    X_train_top = X_train[top_50_features]
    X_test_top = X_test[top_50_features]

    best_model = optimize_hyperparameters(X_train_top, y_train)

    final_model, feature_importances = train_and_evaluate_model(X_train_top, X_test_top, y_train, y_test, best_model)

    print("\nResumen del modelo:")
    print(f"Número de características utilizadas: {len(top_50_features)}")
    print(f"Tamaño del conjunto de entrenamiento: {X_train_top.shape}")
    print(f"Tamaño del conjunto de prueba: {X_test_top.shape}")
    print("\nMejores hiperparámetros:")
    for param, value in best_model.get_params().items():
        print(f"{param}: {value}")

    # Exportar el modelo
    output_dir = '/resultado'
    os.makedirs(output_dir, exist_ok=True)
    model_filename = f'modelo_{category}.pickle'
    with open(os.path.join(output_dir, model_filename), 'wb') as f:
        pickle.dump(final_model, f)
    print(f"\nModelo guardado en: {os.path.join(output_dir, model_filename)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == '__main__':
    # Cambia 'venta' por 'alquiler' si quieres entrenar el modelo para alquiler
    main(category='venta')
