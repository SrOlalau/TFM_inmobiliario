import psycopg2
import pandas as pd
import requests
import time
import json
import os
import re
import pgeocode
from psycopg2 import sql
from fuzzywuzzy import fuzz

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

def send_telegram_message(message):
    """
    Envía un mensaje a través de Telegram utilizando el bot y chat ID proporcionados.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Error sending message: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def connect_db(dbname, user, password, host, port):
    """
    Establece una conexión a la base de datos PostgreSQL.
    """
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def load_data_from_db(conn, query):
    """
    Carga los datos desde la base de datos utilizando la conexión proporcionada y una consulta SQL.
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def get_existing_records(conn_target):
    """
    Obtiene una lista de combinaciones de href y fecha_extract ya insertadas en la tabla 'datos_limpios_con_geo' de la base de datos de destino.
    """
    query = "SELECT href, fecha_extract FROM datos_limpios_con_geo"
    existing_records = pd.read_sql_query(query, conn_target)
    return set(existing_records.apply(lambda row: (row['href'], row['fecha_extract']), axis=1).tolist())

def check_and_create_table(conn):
    """
    Comprueba si la tabla 'datos_limpios_con_geo' existe. Si no existe, la crea.
    """
    table_name = "datos_limpios_con_geo"
    
    cur = conn.cursor()
    
    cur.execute(f"""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = '{table_name}'
    );
    """)
    
    exists = cur.fetchone()[0]
    
    if not exists:
        cur.execute(f"""
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            precio BIGINT,
            sub_descr TEXT,
            href TEXT,
            ubicacion TEXT,
            habitaciones INT,
            banios INT,
            mt2 BIGINT,
            planta TEXT,
            publicado_hace TEXT,
            plataforma TEXT,
            origen TEXT,
            otros TEXT,
            latitude FLOAT,
            longitude FLOAT,
            raw_json TEXT,
            ccaa TEXT,
            fuente_datos TEXT,
            alquiler_venta TEXT,
            fecha_extract DATE,
            geocoding_error TEXT
        );
        """)
        conn.commit()

    cur.close()

def insert_row_to_db(conn, row):
    """
    Inserta una sola fila en la tabla 'datos_limpios_con_geo'.
    """
    table_name = "datos_limpios_con_geo"
    
    cur = conn.cursor()

    columns = [
        'precio', 'sub_descr', 'href', 'ubicacion', 'habitaciones', 'banios', 'mt2', 
        'planta', 'publicado_hace', 'plataforma', 'origen', 'otros', 'latitude', 
        'longitude', 'raw_json', 'ccaa', 'fuente_datos', 'alquiler_venta', 
        'fecha_extract', 'geocoding_error'
    ]
    
    values = [
        row['precio'], row['sub_descr'], row['href'], row['ubicacion'], 
        row['habitaciones'] if pd.notna(row['habitaciones']) else None, 
        row['banios'] if pd.notna(row['banios']) else None, 
        row['mt2'] if pd.notna(row['mt2']) else None, row['planta'], 
        row['publicado_hace'], row['plataforma'], row['origen'], row['otros'], 
        row['latitude'] if pd.notna(row['latitude']) else None, 
        row['longitude'] if pd.notna(row['longitude']) else None, 
        row['raw_json'], row['ccaa'], row['fuente_datos'], 
        row['alquiler_venta'], row['fecha_extract'], row['geocoding_error'] if pd.notna(row['geocoding_error']) else None
    ]

    try:
        insert_query = sql.SQL("""
        INSERT INTO {table} ({fields}) VALUES ({placeholders})
        """).format(
            table=sql.Identifier(table_name),
            fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(', ').join(sql.Placeholder() * len(columns))
        )
        cur.execute(insert_query, values)
        conn.commit()
    
    except Exception as e:
        conn.rollback()
    
    cur.close()

class GeocoderCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.cache_keys = set(self.cache.keys())

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_coordinates(self, address):
        if address in self.cache_keys:
            cached_result = self.cache[address]
            if isinstance(cached_result, dict) and 'error' in cached_result:
                return None, None, False
            return cached_result[0], cached_result[1], False

        try:
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
            headers = {'User-Agent': 'YourAppName/1.0 (your@email.com)'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data:
                lat, lon = data[0]['lat'], data[0]['lon']
                if self.is_valid_coordinates(lat, lon):
                    self.cache[address] = (lat, lon)
                    return lat, lon, True
                else:
                    error_msg = "Coordinates out of expected range"
                    self.cache[address] = {'error': error_msg}
                    return None, None, True
            else:
                error_msg = "No results found"
                self.cache[address] = {'error': error_msg}
                return None, None, True
        except requests.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            self.cache[address] = {'error': error_msg}
            return None, None, True
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.cache[address] = {'error': error_msg}
            return None, None, True

    def is_valid_coordinates(self, lat, lon):
        try:
            lat = float(lat)
            lon = float(lon)
            if 36.0 <= lat <= 43.0 and -9.0 <= lon <= 4.0:
                return True
            return False
        except ValueError:
            return False

    def process_and_geocode(self, df, address_column='ubicacion'):
        mask = df['latitude'].isna() | df['longitude'].isna()
        addresses_to_geocode = df.loc[mask, address_column].dropna().unique()

        for index, address in enumerate(addresses_to_geocode):
            lat, lon, request_made = self.get_coordinates(address)

            if lat is not None and lon is not None:
                df.loc[(df[address_column] == address) & mask, 'latitude'] = float(lat)
                df.loc[(df[address_column] == address) & mask, 'longitude'] = float(lon)
                df.loc[(df[address_column] == address) & mask, 'geocoding_error'] = None  # Limpiar el error si se encontró
            else:
                error = self.cache[address].get('error') if isinstance(self.cache[address], dict) else None
                df.loc[(df[address_column] == address) & mask, 'geocoding_error'] = error

            if request_made:
                time.sleep(1)

            # Mostrar progreso en la consola
            print(f"Geocoded {index + 1} of {len(addresses_to_geocode)} addresses.")

        self.save_cache()
        return df

    def filter_and_geocode_errors(self, error_addresses, df, address_column='ubicacion'):
        """
        Intenta geocodificar direcciones fallidas utilizando el código postal.
        """
        postal_code_regex = re.compile(r'^(\d{5}),')

        postal_codes = []
        address_map = {}

        # Extraer códigos postales de las direcciones fallidas
        for address in error_addresses:
            if address and isinstance(address, str):
                match = postal_code_regex.match(address)
                if match:
                    postal_code = match.group(1)
                    postal_codes.append(postal_code)
                    address_map[postal_code] = address

        # Consultar pgeocode para los códigos postales
        nomi = pgeocode.Nominatim('es')
        results = nomi.query_postal_code(postal_codes)

        for i, result in results.iterrows():
            if not pd.isna(result['latitude']) and not pd.isna(result['longitude']):
                postal_code = result['postal_code']
                lat, lon = result['latitude'], result['longitude']
                original_address = address_map[postal_code]

                # Fuzzy matching para validar la coincidencia
                relevant_fields = [
                    result['place_name'],
                    result['state_name'],
                    result['community_name'],
                    result['county_name']
                ]

                address_text = postal_code_regex.sub('', original_address).strip().lower()

                valid_match = False
                for field in relevant_fields:
                    if pd.notna(field):
                        field = field.lower()
                        score = fuzz.token_set_ratio(field, address_text)
                        if score >= 97:
                            valid_match = True
                            break

                if valid_match:
                    self.cache[original_address] = (lat, lon)
                    df.loc[(df[address_column] == original_address), 'latitude'] = float(lat)
                    df.loc[(df[address_column] == original_address), 'longitude'] = float(lon)
                    df.loc[(df[address_column] == original_address), 'geocoding_error'] = None
                else:
                    print(f"No valid match found for {original_address} with postal code {postal_code}. Skipping.")

        self.save_cache()

        return df

def main():
    send_telegram_message("El proceso de geocode ha iniciado.")
    
    # Conectar a la base de datos de origen
    conn_source = connect_db("datos_limpios", "datos_limpios", "datos_limpios", "10.1.2.2", "5439")
    if conn_source is None:
        return

    # Conectar a la base de datos de destino
    conn_target = connect_db("geoloc", "geoloc", "geoloc", "10.1.2.2", "5441")
    if conn_target is None:
        return
    
    # Comprobar si la tabla existe y crearla si es necesario
    check_and_create_table(conn_target)

    # Obtener combinaciones de href y fecha_extract ya insertadas
    existing_records = get_existing_records(conn_target)
    
    # Definir la consulta SQL para obtener los datos
    query = "SELECT * FROM consolidated_data"
    
    # Cargar los datos desde la base de datos de origen
    df = load_data_from_db(conn_source, query)
    
    if df is None or df.empty:
        return
    
    # Cerrar la conexión a la base de datos de origen
    conn_source.close()
    
    # Filtrar filas nuevas (combinaciones de href y fecha_extract no en existing_records)
    new_rows = df[~df.apply(lambda row: (row['href'], row['fecha_extract']), axis=1).isin(existing_records)]
    
    if new_rows.empty:
        send_telegram_message("No hay nuevas filas para procesar.")
        return
    
    # Establecer el nombre del archivo de caché en la misma ruta que el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, 'geocode_cache.json')
    
    # Inicializar la caché de geocodificación
    geocoder = GeocoderCache(cache_file=cache_path)
    
    # Procesar y geocodificar las filas sin longitud ni latitud
    result_df = geocoder.process_and_geocode(new_rows, address_column='ubicacion')
    
    # Identificar direcciones con errores en la geocodificación
    error_mask = result_df['latitude'].isna() | result_df['longitude'].isna() | result_df['geocoding_error'].notna()
    error_addresses = result_df.loc[error_mask, 'ubicacion'].unique()

    if len(error_addresses) > 0:
        result_df = geocoder.filter_and_geocode_errors(error_addresses, result_df, address_column='ubicacion')
    
    # Insertar todas las filas (incluyendo las que ya tenían longitud y latitud)
    for index, row in new_rows.iterrows():
        insert_row_to_db(conn_target, row)
        print(f"Inserted {index + 1} of {len(new_rows)} rows into database.")
    
    # Cerrar la conexión a la base de datos de destino
    conn_target.close()

    send_telegram_message(f"El proceso de geocode ha finalizado. Filas insertadas: {len(new_rows)}")

# Ejecutar el script
main()
