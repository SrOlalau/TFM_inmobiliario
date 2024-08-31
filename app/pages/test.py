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
import streamlit as st
from app.models.load_models import load_model

# Cargar el modelo de machine learning
model = load_model()

# Título de la aplicación
st.title('Predicción de Precios de Habitaciones en España')

# Entrada de parámetros del usuario
habitaciones = st.number_input('Número de habitaciones', min_value=1, max_value=10, value=3)
banios = st.number_input('Número de baños', min_value=1, max_value=5, value=2)
mt2 = st.number_input('Metros cuadrados', min_value=30, max_value=2500, value=100)
longitude = st.number_input('longitude', value=-3.7038)
latitude = st.number_input('latitude', value=40.4168)
publicado_hace = 0
planta = 0

# Botón para realizar la predicción
if st.button('Predecir Precio'):
    # Crear un DataFrame con los parámetros
    input_data = pd.DataFrame({
    'banios': [banios],
    'publicado_hace': [publicado_hace],
    'longitude': [longitude],
    'planta': [planta],
    'latitude': [latitude],
    'mt2': [mt2],
    'habitaciones' : [habitaciones]

    # Agregar todas las demás columnas necesarias
})
    
    # Realizar la predicción
    prediccion = model.predict(input_data)
    
    # Mostrar el resultado
    st.write(f'El precio estimado de la habitación es: {prediccion[0]:,.2f} €')

# Visualización de un mapa (opcional)
st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
