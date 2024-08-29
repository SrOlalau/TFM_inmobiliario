import pandas as pd
import requests
import time
import json
import os
import re
import pgeocode
from fuzzywuzzy import fuzz
import psycopg2
from psycopg2 import sql

def connect_db(db_name, db_user, db_password, db_host, db_port):
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    return conn

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
                print(f"Warning: Cache file {self.cache_file} is corrupted. Starting with an empty cache.")
                return {}
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_coordinates(self, address):
        if address in self.cache_keys:
            cached_result = self.cache[address]
            if isinstance(cached_result, dict) and 'error' in cached_result:
                print(f"Skipping {address} due to previous error: {cached_result['error']}")
                return None, None, False
            return cached_result[0], cached_result[1], False

        try:
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data:
                lat, lon = data[0]['lat'], data[0]['lon']
                self.cache[address] = (lat, lon)
                return lat, lon, True
            else:
                error_msg = "No results found"
                self.cache[address] = {'error': error_msg}
                print(f"Error geocoding {address}: {error_msg}")
                return None, None, True
        except requests.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            self.cache[address] = {'error': error_msg}
            print(f"Error geocoding {address}: {error_msg}")
            return None, None, True
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.cache[address] = {'error': error_msg}
            print(f"Error geocoding {address}: {error_msg}")
            return None, None, True

    def process_and_geocode(self, df, address_column='ubicacion'):
        mask = df['latitude'].isna() | df['longitude'].isna()
        addresses_to_geocode = df.loc[mask, address_column].dropna().unique()

        for address in addresses_to_geocode:
            lat, lon, request_made = self.get_coordinates(address)

            if lat is not None and lon is not None:
                df.loc[(df[address_column] == address) & mask, 'latitude'] = float(lat)
                df.loc[(df[address_column] == address) & mask, 'longitude'] = float(lon)
            else:
                error = self.cache[address].get('error') if isinstance(self.cache[address], dict) else None
                df.loc[(df[address_column] == address) & mask, 'geocoding_error'] = error

            if request_made:
                time.sleep(1)  # Pause only if a new request was made

        self.save_cache()
        return df

    def filter_and_geocode_errors(self, error_addresses, df, address_column='ubicacion'):
        postal_code_regex = re.compile(r'^(\d{5}),')
        postal_codes = []
        address_map = {}

        for address in error_addresses:
            if address and isinstance(address, str):
                match = postal_code_regex.match(address)
                if match:
                    postal_code = match.group(1)
                    postal_codes.append(postal_code)
                    address_map[postal_code] = address
            else:
                print(f"Skipping invalid address: {address}")

        nomi = pgeocode.Nominatim('es')
        results = nomi.query_postal_code(postal_codes)

        for i, result in results.iterrows():
            if not pd.isna(result['latitude']) and not pd.isna(result['longitude']):
                postal_code = result['postal_code']
                lat, lon = result['latitude'], result['longitude']
                original_address = address_map[postal_code]

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
                else:
                    print(f"No valid match found for {original_address} with postal code {postal_code}. Skipping.")

        self.save_cache()
        return df

def create_table_if_not_exists(conn, table_name):
    with conn.cursor() as cur:
        cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
        """, (table_name,))
        table_exists = cur.fetchone()[0]

        if not table_exists:
            cur.execute(sql.SQL("""
            CREATE TABLE {} (
                id integer,
                precio bigint,
                sub_descr text,
                href text,
                ubicacion text,
                habitaciones integer,
                banios integer,
                m2 bigint,
                planta text,
                publicado_hace text,
                plataforma text,
                origen text,
                otros text,
                latitude double precision,
                longitude double precision,
                raw_json text,
                ccaa text,
                fuente_datos text,
                alquiler_venta text,
                fecha_extract date,
                geocoding_error text
            )
            """).format(sql.Identifier(table_name)))
            conn.commit()

def process_batch(batch_df, geocoder, conn_target, table_name):
    # Process and geocode the batch
    result_df = geocoder.process_and_geocode(batch_df, address_column='ubicacion')

    # Identify addresses with errors in geocoding
    error_mask = result_df['latitude'].isna() | result_df['longitude'].isna() | result_df['geocoding_error'].notna()
    error_addresses = result_df.loc[error_mask, 'ubicacion'].unique()

    # Filter and geocode these error addresses using the pgeocode library
    if len(error_addresses) > 0:
        result_df = geocoder.filter_and_geocode_errors(error_addresses, result_df, address_column='ubicacion')

    # Insert data into target database
    with conn_target.cursor() as cur:
        for _, row in result_df.iterrows():
            cur.execute(sql.SQL("""
            INSERT INTO {} (
                id, precio, sub_descr, href, ubicacion, habitaciones, banios, m2, planta,
                publicado_hace, plataforma, origen, otros, latitude, longitude, raw_json,
                ccaa, fuente_datos, alquiler_venta, fecha_extract, geocoding_error
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """).format(sql.Identifier(table_name)), tuple(row))

    conn_target.commit()
    print(f"Processed and inserted batch of {len(result_df)} rows")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, 'geocode_cache.json')
    geocoder = GeocoderCache(cache_file=cache_path)

    # Connect to source and target databases
    conn_source = connect_db("datos_limpios", "datos_limpios", "datos_limpios", "10.1.2.2", "5439")
    conn_target = connect_db("geoloc", "geoloc", "geoloc", "10.1.2.2", "5441")

    original_table_name = "consolidated_data"
    new_table_name = "tabla_geocodificada"

    # Create table in target database if it doesn't exist
    create_table_if_not_exists(conn_target, new_table_name)

    # Process data in batches
    batch_size = 1000
    offset = 0

    while True:
        # Read data from source database in batches
        with conn_source.cursor() as cur:
            cur.execute(sql.SQL("""
                SELECT * FROM {} 
                ORDER BY id
                LIMIT %s OFFSET %s
            """).format(sql.Identifier(original_table_name)), (batch_size, offset))
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()

        if not data:
            break  # No more data to process

        batch_df = pd.DataFrame(data, columns=columns)
        process_batch(batch_df, geocoder, conn_target, new_table_name)

        offset += batch_size
        print(f"Processed {offset} rows so far")

    # Close database connections
    conn_source.close()
    conn_target.close()

    print("Geocoding process completed and all data saved to target database.")

if __name__ == "__main__":
    main()