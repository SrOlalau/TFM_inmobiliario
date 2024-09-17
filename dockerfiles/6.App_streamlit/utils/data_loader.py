import pickle

def cargar_modelo(ruta_modelo):
    with open(ruta_modelo, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # Como loaded_data es un objeto RandomForestRegressor, lo retornamos como el modelo
    print(f"Tipo de loaded_data: {type(loaded_data)}")  # Solo para verificar el tipo
    
    # Retornar el modelo cargado (pipeline) y un valor placeholder para features
    return loaded_data, None  # En este caso, no tienes un 'features', ajusta según lo que necesites
