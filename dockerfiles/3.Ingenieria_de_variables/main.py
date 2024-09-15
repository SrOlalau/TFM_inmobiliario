import psycopg2
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
import numpy as np
import requests

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

def connect_db(db_name, db_user, db_password, db_host, db_port):
    """
    Establece una conexión a la base de datos PostgreSQL.
    """
    return psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )

def load_poi_data(db_name, db_user, db_password, db_host, db_port, table_name):
    conn = connect_db(db_name, db_user, db_password, db_host, db_port)
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_coordinates_data(db_name, db_user, db_password, db_host, db_port, table_name):
    conn = connect_db(db_name, db_user, db_password, db_host, db_port)
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_existing_records(conn, table_name):
    """
    Verifica si la tabla existe y, si es así, obtiene una lista de combinaciones de href y fecha_extract
    ya insertadas en la tabla de destino.
    """
    try:
        query = f"SELECT href, fecha_extract FROM {table_name};"
        existing_records = pd.read_sql_query(query, conn)
        return set(existing_records.apply(lambda row: (row['href'], row['fecha_extract']), axis=1).tolist())
    except Exception as e:
        print(f"La tabla {table_name} no existe o ocurrió un error al verificar registros existentes: {e}")
        return None

def filter_new_data(df, existing_records):
    """
    Filtra los registros que ya existen en la tabla de destino.
    """
    if existing_records is None:
        return df  # Si no hay registros existentes, devuelve todo el DataFrame
    return df[~df.apply(lambda row: (row['href'], row['fecha_extract']), axis=1).isin(existing_records)]

class POI_counter:
    def __init__(self, poi_df):
        if 'type' not in poi_df.columns:
            raise KeyError("La columna 'type' no existe en los datos de POIs.")
        self.poi_df = poi_df
        self.filtered_poi_df = self._filter_poi_types()
        self.poi_types = self.filtered_poi_df['type'].dropna().unique()
        self.selected_types = self.poi_types
        self.radii = [1]  # Default radius
        self.df_coords = None
        self.closest_types = []

    def _filter_poi_types(self, min_count=20):
        type_counter = Counter(self.poi_df['type'])
        return self.poi_df[self.poi_df['type'].map(type_counter) >= min_count]

    def set_poi_types(self, types):
        if not set(types).issubset(self.poi_types):
            raise ValueError("Some of the provided types are not valid.")
        self.selected_types = types

    def set_radii(self, radii):
        self.radii = radii

    @staticmethod
    def _squared_euclidean_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        x = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
        y = lat2 - lat1
        return (R * (x ** 2 + y ** 2) * np.pi / 180) ** 2

    @staticmethod
    def _bounding_box_filter(center, pois, radius):
        lat, lon = center
        delta = radius / 111
        lat_min, lat_max = lat - delta, lat + delta
        lon_min, lon_max = lon - delta, lon + delta
        return pois[(pois[:, 0] >= lat_min) & (pois[:, 0] <= lat_max) &
                    (pois[:, 1] >= lon_min) & (pois[:, 1] <= lon_max)]

    @staticmethod
    def _count_pois_within_radius(center, pois, radius):
        filtered_pois = POI_counter._bounding_box_filter(center, pois, radius)
        if len(filtered_pois) == 0:
            return 0
        squared_distances = POI_counter._squared_euclidean_distance(center[0], center[1], filtered_pois[:, 0],
                                                                    filtered_pois[:, 1])
        return np.sum(squared_distances <= radius ** 2)

    @staticmethod
    def _count_pois_for_multiple_centers(pois, centers, radius):
        return Parallel(n_jobs=-1)(
            delayed(POI_counter._count_pois_within_radius)(center, pois, radius) for center in centers)

    @staticmethod
    def _find_closest_poi(center, pois):
        if len(pois) == 0:
            return np.inf
        distances = POI_counter._squared_euclidean_distance(center[0], center[1], pois[:, 0], pois[:, 1])
        return np.sqrt(np.min(distances))

    def _find_closest_pois_for_multiple_centers(self, pois, centers):
        return Parallel(n_jobs=-1)(delayed(self._find_closest_poi)(center, pois) for center in centers)

    def calculate_poi_counts(self, df):
        result_df = df.copy()
        if self.df_coords is None:
            self.na_mask = result_df[['latitude', 'longitude']].isna().any(axis=1)
            self.df_coords = result_df.dropna(subset=['latitude', 'longitude'])[['latitude', 'longitude']].to_numpy()
        new_columns = {}
        for radius in self.radii:
            for poi_type in self.selected_types:
                poi_type_df = self.filtered_poi_df[self.filtered_poi_df['type'] == poi_type]
                pois = poi_type_df[['lat', 'lon']].to_numpy()
                poi_counts = self._count_pois_for_multiple_centers(pois, self.df_coords, radius)
                col_name = f'POI_{poi_type}_{radius}km'
                new_columns[col_name] = pd.Series(poi_counts, index=result_df.index[~self.na_mask])

        for poi_type in self.closest_types:
            poi_type_df = self.filtered_poi_df[self.filtered_poi_df['type'] == poi_type]
            pois = poi_type_df[['lat', 'lon']].to_numpy()
            closest_distances = self._find_closest_pois_for_multiple_centers(pois, self.df_coords)
            col_name = f'closest_{poi_type}'
            new_columns[col_name] = pd.Series(closest_distances, index=result_df.index[~self.na_mask])

        new_columns_df = pd.DataFrame(new_columns)
        result_df = pd.concat([result_df, new_columns_df], axis=1)
        for col in new_columns_df.columns:
            result_df.loc[self.na_mask, col] = np.nan
        return result_df

def create_table_and_insert_data(df, db_name, db_user, db_password, db_host, db_port, table_name):
    conn = connect_db(db_name, db_user, db_password, db_host, db_port)
    cursor = conn.cursor()

    # Definir las columnas de la tabla
    column_definitions = [
        "id SERIAL PRIMARY KEY",
        "precio FLOAT",
        "sub_descr TEXT",
        "href TEXT",
        "ubicacion TEXT",
        "habitaciones FLOAT",
        "banios FLOAT",
        "mt2 FLOAT",
        "planta FLOAT",
        "publicado_hace TEXT",
        "plataforma TEXT",
        "origen TEXT",
        "otros TEXT",
        "latitude FLOAT",
        "longitude FLOAT",
        "raw_json TEXT",
        "ccaa TEXT",
        "fuente_datos TEXT",
        "alquiler_venta TEXT",
        "fecha_extract DATE",
        "geocoding_error TEXT"
    ]

    # Agregar las columnas de POIs
    poi_columns = [f"{col} FLOAT" for col in df.columns if col.startswith('POI_')]
    column_definitions.extend(poi_columns)

    # Crear la tabla
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(column_definitions)}
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

    # Insertar los datos en la tabla
    for i, row in df.iterrows():
        placeholders = ', '.join(['%s'] * len(row))
        columns = ', '.join(row.index)
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

def main():
    # Enviar mensaje al iniciar el proceso
    send_telegram_message("El proceso de cálculo de POI ha comenzado.")

    db_name_poi = "scraping_openstreetmaps"
    db_user_poi = "POI"
    db_password_poi = "POI"
    db_host_poi = "10.1.2.2"
    db_port_poi = "5438"
    table_name_poi = "points_of_interest"

    poi_df = load_poi_data(db_name_poi, db_user_poi, db_password_poi, db_host_poi, db_port_poi, table_name_poi)

    db_name_coords = "geoloc"
    db_user_coords = "geoloc"
    db_password_coords = "geoloc"
    db_host_coords = "10.1.2.2"
    db_port_coords = "5441"
    table_name_coords = "datos_limpios_con_geo"

    # Cargar datos de coordenadas
    coords_df = load_coordinates_data(db_name_coords, db_user_coords, db_password_coords, db_host_coords, db_port_coords, table_name_coords)

    # Conectar a la base de datos de destino para obtener registros existentes
    conn_target = connect_db("geo_y_poi", "geo_y_poi", "geo_y_poi", "10.1.2.2", "5442")
    existing_records = get_existing_records(conn_target, "Datos_limpios_con_geo_y_poi")
    conn_target.close()

    # Filtrar solo las filas nuevas que no están en la tabla de destino
    new_coords_df = filter_new_data(coords_df, existing_records)

    if new_coords_df.empty:
        send_telegram_message("No hay datos nuevos para procesar.")
        return

    poi_counter = POI_counter(poi_df)
    poi_counter.set_poi_types(poi_counter.poi_types)
    poi_counter.set_radii([1, 3, 5])

    resultado_df = poi_counter.calculate_poi_counts(new_coords_df)

    create_table_and_insert_data(resultado_df, "geo_y_poi", "geo_y_poi", "geo_y_poi", "10.1.2.2", "5442", "Datos_limpios_con_geo_y_poi")

    # Enviar mensaje al finalizar el proceso con el número de filas procesadas
    send_telegram_message(f"El proceso de cálculo de POI ha terminado. Se han procesado {len(resultado_df)} filas.")

if __name__ == "__main__":
    main()
