import pickle
import streamlit as st

@st.cache_resource
def cargar_modelo(path):
    """
    Carga el modelo y las características desde un archivo pickle.
    
    Args:
    path (str): La ruta del archivo pickle.

    Returns:
    tuple: Un pipeline de modelo y las características.
    """
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data['pipeline'], loaded_data['features']
