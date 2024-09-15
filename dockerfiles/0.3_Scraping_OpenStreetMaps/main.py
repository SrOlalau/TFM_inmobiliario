import requests
import pandas as pd
import os
import psycopg2
from psycopg2 import sql
import time
import random

# Configura tu token de bot y el ID de chat
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

# Variables de conexión a la base de datos
DB_NAME = "scraping_openstreetmaps"
DB_USER = "POI"
DB_PASSWORD = "POI"
DB_HOST = "10.1.2.2"
DB_PORT = "5438"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print(f"Error enviando mensaje a Telegram: {response.status_code}")

def get_first_existing_key(tags, POI):
    for key in POI:
        if key in tags:
            return tags[key]
    return None

def osm_downloader(location):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    area[name="{location}"]->.searchArea;
    (
      node(area.searchArea)["shop"="supermarket"];
      node(area.searchArea)["shop"="convenience"];
      node(area.searchArea)["shop"="marketplace"];
      node(area.searchArea)["amenity"="pharmacy"];
      node(area.searchArea)["amenity"="hospital"];
      node(area.searchArea)["amenity"="clinic"];
      node(area.searchArea)["amenity"="doctors"];
      node(area.searchArea)["amenity"="health"];
      node(area.searchArea)["amenity"="gym"];
      node(area.searchArea)["amenity"="school"];
      node(area.searchArea)["amenity"="college"];
      node(area.searchArea)["amenity"="library"];
      node(area.searchArea)["public_transport"="station"];
      node(area.searchArea)["highway"="bus_stop"];
      node(area.searchArea)["railway"="station"];
      node(area.searchArea)["amenity"="bicycle_rental"];
      node(area.searchArea)["leisure"="park"];
      node(area.searchArea)["leisure"="garden"];
      node(area.searchArea)["leisure"="playground"];
      node(area.searchArea)["leisure"="sports_centre"];
      node(area.searchArea)["shop"="mall"];
      node(area.searchArea)["shop"="clothes"];
      node(area.searchArea)["amenity"="cinema"];
      node(area.searchArea)["amenity"="theatre"];
      node(area.searchArea)["amenity"="restaurant"];
      node(area.searchArea)["amenity"="cafe"];
      node(area.searchArea)["amenity"="bar"];
      node(area.searchArea)["tourism"="museum"];
      node(area.searchArea)["historic"="monument"];
      node(area.searchArea)["historic"="memorial"];
      node(area.searchArea)["tourism"="viewpoint"];
      node(area.searchArea)["amenity"="post_office"];
      node(area.searchArea)["amenity"="police"];
      node(area.searchArea)["amenity"="fire_station"];
      node(area.searchArea)["amenity"="townhall"];
      node(area.searchArea)["amenity"="waste_disposal"];
      node(area.searchArea)["landuse"="landfill"];
      node(area.searchArea)["landuse"="industrial"];
      node(area.searchArea)["amenity"="prison"];
      node(area.searchArea)["man_made"="works"];
      node(area.searchArea)["amenity"="factory"];
      node(area.searchArea)["amenity"="atm"];
      node(area.searchArea)["aeroway"="aerodrome"];
    );
    out center;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    POI = ['amenity', 'shop', 'leisure', 'public_transport', 'tourism', 'historic', 'landuse', 'man_made', 'aeroway']

    points_of_interest = []
    for element in data['elements']:
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
        elif 'center' in element:
            lat = element['center']['lat']
            lon = element['center']['lon']
        else:
            continue

        tags = element.get('tags', {})
        poi_type = get_first_existing_key(tags, POI) if tags else 'N/A'
        poi_name = tags.get('name', 'Unnamed')
        poi_tags = ', '.join([f"{k}: {v}" for k, v in tags.items()])

        points_of_interest.append([poi_type, poi_name, lat, lon, poi_tags])

    df = pd.DataFrame(points_of_interest, columns=['type', 'name', 'lat', 'lon', 'tags'])
    return data, df

def connect_db():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def create_table_if_not_exists(conn):
    with conn.cursor() as cur:
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS points_of_interest (
                id SERIAL PRIMARY KEY,
                type TEXT,
                name TEXT,
                lat FLOAT,
                lon FLOAT,
                tags TEXT,
                region TEXT,
                date_added DATE DEFAULT CURRENT_DATE
            )
        """)
        cur.execute(create_table_query)
        conn.commit()

def insert_data_into_db(conn, df, region):
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            insert_query = sql.SQL("""
                INSERT INTO points_of_interest (type, name, lat, lon, tags, region)
                VALUES (%s, %s, %s, %s, %s, %s)
            """)
            cur.execute(insert_query, (row['type'], row['name'], row['lat'], row['lon'], row['tags'], region))
        conn.commit()

def main():
    send_telegram_message("Iniciando el script de descarga de datos OSM.")
    
    conn = connect_db()
    create_table_if_not_exists(conn)

    ccaa = [
        'Madrid', 'Catalunya', 'València', 'Granada',
        'Málaga', 'Sevilla', 'Cádiz', 'Cantabria', 'Euskadi'
    ]

    try:
        for region in ccaa:
            print(f'Realizando consulta a OSM, via Overpass API: {region}')
            send_telegram_message(f"Iniciando descarga de datos para {region}")
            
            data, df = osm_downloader(region)
            insert_data_into_db(conn, df, region)
            
            send_telegram_message(f"Datos de {region} descargados y guardados en la base de datos.")
            
            # Espera aleatoria para evitar sobrecarga del servidor
            time.sleep(random.uniform(5, 15))

    except Exception as e:
        error_message = f"Error en el script: {str(e)}"
        print(error_message)
        send_telegram_message(error_message)
    finally:
        conn.close()

    send_telegram_message("El script de descarga de datos OSM ha finalizado.")

if __name__ == '__main__':
    main()