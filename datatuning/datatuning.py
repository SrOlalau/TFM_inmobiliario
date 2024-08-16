import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic

def calculate_distance(coord1, coord2):
    if pd.isnull(coord1) or pd.isnull(coord2):
        return float('inf')  # Retorna infinito si alguna coordenada es NaN
    return geodesic(coord1, coord2).meters

def main(script_dir):
    output_path = os.path.join(script_dir, 'datatuning/datatuning.csv')
    origin_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    points_of_interest = os.path.join(script_dir, 'data/POI/points_of_interest_ES.csv')
    
    # Cargar dataframes
    poi_df = pd.read_csv(points_of_interest)
    df = pd.read_csv(origin_path)
    
    # Lista de tipos de POIs
    poi_types = ['aerodrome','atm','bar','bicycle_rental','bus_stop','cafe','cinema',
                 'clinic','clothes','college','convenience','doctors','factory',
                 'fire_station','garden','gym','hospital','health','industrial',
                 'landfill','library','mall','marketplace','memorial','monument',
                 'museum','park','pharmacy','playground','police','post_office',
                 'prison','restaurant','school','sports_centre','station','theatre',
                 'townhall','viewpoint','waste_disposal','works']
    
    # Asegúrate de que las coordenadas de los POIs sean tuplas de (lat, lon)
    poi_df['coordinates'] = poi_df[['lat', 'lon']].apply(lambda row: (row['lat'], row['lon']), axis=1)
    
    # Inicializa las columnas para los POIs con cadenas vacías
    df[poi_types] = ''
    
    distance_a_poi = 1000  # Distancia en metros
    
    # Procesa cada fila del dataframe df
    for index, row in df.iterrows():
        if pd.isnull(row['latitude']) or pd.isnull(row['longitude']):
            continue  # Ignorar filas con coordenadas NaN
        
        location = (row['latitude'], row['longitude'])
        
        # Procesa cada tipo de POI
        for poi_type in poi_types:
            # Filtra POIs por tipo
            filtered_pois = poi_df[poi_df['type'] == poi_type]
            
            # Encuentra los POIs cercanos
            nearby_pois = filtered_pois[filtered_pois['coordinates'].apply(lambda x: calculate_distance(location, x) <= distance_a_poi)]
            
            # Actualiza el dataframe con la información de POIs encontrados
            if not nearby_pois.empty:
                # Convierte la lista de nombres a una cadena separada por comas
                df.at[index, poi_type] = ', '.join(nearby_pois['name'].tolist())

    # Guarda el dataframe actualizado a un nuevo archivo CSV
    df.to_csv(output_path, index=False)