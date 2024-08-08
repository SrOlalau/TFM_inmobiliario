# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:57:51 2023

@author: avg
"""
import html
import json
import math
import os
import random
import re
import time
from datetime import date
from json.decoder import JSONDecodeError

import pandas as pd
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import sql

# Configura tu token de bot y el ID de chat
TELEGRAM_BOT_TOKEN = '6916058231:AAEOmgGX0k427p5mbe6UFmxAL1MpTXYCYTs'
TELEGRAM_CHAT_ID = '297175679'

# Variables de conexión a la base de datos
DB_NAME = "trovit_scraping"
DB_USER = "trovit"
DB_PASSWORD = "trovit"
DB_HOST = "10.1.5.9"
DB_PORT = "5432"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print(f"Error enviando mensaje a Telegram: {response.status_code}")

# Enviar mensaje al iniciar
send_telegram_message("Iniciando el script de scraping de pisos.")

# Aquí se inserta el segundo código para los User Agents y la función user_agents()
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
    'Mozilla/5.0 (iPad; CPU OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
    'Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.66 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 13_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36',
    'Mozilla/5.0 (Linux; Android 10; SM-G950U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Mobile Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
    'Mozilla/5.0 (iPad; CPU OS 13_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/606.5.15 (KHTML, like Gecko) Version/11.1.2 Safari/606.5.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0',
    'Mozilla/5.0 (Linux; Android 8.0.0; SM-G930F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/601.6.17 (KHTML, like Gecko) Version/11.0.3 Safari/601.6.17'
]

# Creación de directorios
def crear_directorios(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    raw_data_path = os.path.join(base_path, "RawDataTFM")
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    return raw_data_path

# Cambiar la base_path para que apunte al nivel de main.py
base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
raw_data_path = crear_directorios(base_path)

# Returns the current local date
today = date.today()

# Calcula el número de páginas para iterar
def num_paginas(soup):    
    n_resultados = soup.find(class_ = 'grid__title')
    if n_resultados and n_resultados.text:
        resultados = re.findall(r'\d+', n_resultados.text.replace('.', ''))
        if resultados:
            n_resultados = int(resultados[0])
            n_paginas = math.ceil(n_resultados / 30)
            return n_paginas if n_paginas else None
    return 0  # Si no hay resultados, devuelve 0 páginas

def extract_property_info(cuadro_info):
    property_info = {}

    # Extract price
    if cuadro_info.span:
        property_info["precio"] = cuadro_info.span.text.strip()
    
    # Extract sub_description
    if cuadro_info.a:
        property_info["sub_descr"] = cuadro_info.a.text.strip()
        property_info["href"] = cuadro_info.a['href'].strip()

    # Extract location
    location_element = cuadro_info.find("p", class_="p-sm")
    if location_element:
        property_info["ubicacion"] = location_element.text.strip()

    # Extract bedrooms, bathrooms, area, and floor
    details_element = cuadro_info.find_all(class_="ad-preview__char p-sm")
    for detail in details_element:
        detail_text = detail.text.strip()
        
        # Check if detail contains bedroom information
        if re.search(r"hab", detail_text, re.IGNORECASE):
            property_info["habitaciones"] = detail_text
        
        # Check if detail contains bathroom information
        elif re.search(r"bañ[os]", detail_text, re.IGNORECASE):
            property_info["banios"] = detail_text
        
        # Check if detail contains area information
        elif re.search(r"m", detail_text, re.IGNORECASE):
            property_info["mt2"] = detail_text
        
        # Assume detail is floor information
        else:
            property_info["planta"] = detail_text

    # Add additional fields as needed
    description_element = cuadro_info.find(class_="ad-preview__description")
    if description_element:
        property_info["otros"] = description_element.text.strip()

    return property_info

def get_all_relevant(anunc):
    # Primero extrae información del banner
    cuadro_info = anunc.find(class_='ad-preview__bottom').find(class_='ad-preview__info')
    to_save = extract_property_info(cuadro_info)
    
    # Luego coordenadas del script en json
    json_text = anunc.find('script', type='application/ld+json').text
    json_text = html.unescape(json_text)
    json_text = re.sub(r'[\x00-\x1f\x7f]', '', json_text)  # Remove control characters

    try:
        cuadro_inf_json = json.loads(json_text)
        to_save['latitude'] = cuadro_inf_json['geo']['latitude']
        to_save['longitude'] = cuadro_inf_json['geo']['longitude']
    except JSONDecodeError:
        try:
            # Extract latitude and longitude directly using regex
            latitude = re.search(r'"latitude"\s*:\s*(-?\d+\.\d+)', json_text)
            longitude = re.search(r'"longitude"\s*:\s*(-?\d+\.\d+)', json_text)
            
            if latitude and longitude:
                to_save['latitude'] = float(latitude.group(1))
                to_save['longitude'] = float(longitude.group(1))
            else:
                to_save['raw_json'] = json_text  # Save raw JSON if fields are not found
        except Exception as e:
            # Handle any other exception and save the raw JSON
            to_save['raw_json'] = json_text

    return to_save

def hace_busqueda(num_search=None, tipo='alquiler', ccaa='madrid', ultimasemana=False):
    base_url = f'https://www.pisos.com/{tipo}/pisos-{ccaa}/'
    if ultimasemana:
        base_url += 'ultimasemana/'

    url = base_url.format(tipo=tipo, ciudad=ccaa)
    while True:
        try:
            if num_search:
                response = requests.get(url + str(num_search), headers=user_agents(), timeout=20)
            else:
                response = requests.get(url, headers=user_agents(), timeout=20)
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except requests.exceptions.ConnectionError:
            print("Connection error. Waiting for a while before retrying...")
            time.sleep(random.uniform(10, 30))
        except requests.exceptions.ReadTimeout:
            print("Read timeout error. Waiting for a while before retrying...")
            time.sleep(random.uniform(10, 30))

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
            CREATE TABLE IF NOT EXISTS scraping_pisos_tabla (
                id SERIAL PRIMARY KEY,
                precio TEXT,
                sub_descr TEXT,
                href TEXT,
                ubicacion TEXT,
                habitaciones TEXT,
                banios TEXT,
                mt2 TEXT,
                otros TEXT,
                latitude FLOAT,
                longitude FLOAT,
                raw_json TEXT,
                CCAA TEXT,
                fecha DATE DEFAULT CURRENT_DATE
            )
        """)
        cur.execute(create_table_query)
        conn.commit()

def insert_data_into_db(conn, data):
    with conn.cursor() as cur:
        insert_query = sql.SQL("""
            INSERT INTO scraping_pisos_tabla (
                precio, sub_descr, href, ubicacion, habitaciones, banios, mt2, otros,
                latitude, longitude, raw_json, CCAA, fecha
            ) VALUES (
                %(precio)s, %(sub_descr)s, %(href)s, %(ubicacion)s, %(habitaciones)s, %(banios)s, %(mt2)s, %(otros)s,
                %(latitude)s, %(longitude)s, %(raw_json)s, %(CCAA)s, CURRENT_DATE
            )
        """)
        cur.execute(insert_query, data)
        conn.commit()

# Hace las busquedas de ventas y alquileres en madrid
def main(ultimasemana=True):
    send_telegram_message("Iniciando scraping de pisos.com.")
    conn = connect_db()
    create_table_if_not_exists(conn)

    print("Today's date is: ", today)
    TIPOS = ['alquiler', 'venta']
    CCAA = [
        'madrid', 'barcelona', 'valencia', 'granada',
        'malaga', 'sevilla', 'cadiz', 'cantabria', 'vizcaya_bizkaia'
    ]
    random.shuffle(CCAA)

    try:
        for tipo in TIPOS:
            for comunidad in CCAA:
                filename = f'downloaded_data_{tipo}_{comunidad}_{today.year}{today.month:02d}{today.day:02d}.csv'
                filepath = os.path.join(raw_data_path, filename)
                if os.path.exists(filepath):
                    print(f'El archivo {filename} ya existe. Pasando al siguiente archivo.')
                    continue

                print(f'--------- {tipo.upper()} - {comunidad} ---------')
                soup = hace_busqueda(tipo=tipo, ccaa=comunidad, ultimasemana=ultimasemana)
                n_pag = num_paginas(soup)

                if n_pag == 0:
                    print(f"Sin Datos para {tipo} en {comunidad}")
                    continue

                # Generar lista de números de páginas y mezclarla
                paginas = list(range(1, n_pag + 1))
                random.shuffle(paginas)

                data = []
                for idx, i in enumerate(paginas):
                    anuncios = None
                    while anuncios is None:
                        try:
                            print(f'Buscando página {idx + 1}/{n_pag}')
                            soup = hace_busqueda(num_search=i, tipo=tipo, ccaa=comunidad)
                            anuncios = soup.find(class_='grid').find_all(class_='ad-preview')
                        except AttributeError:
                            time.sleep(random.uniform(3, 7))

                    for anunc in anuncios:
                        to_save = get_all_relevant(anunc)
                        if to_save:
                            to_save["CCAA"] = comunidad
                            insert_data_into_db(conn, to_save)
                            data.append(to_save)
                    time.sleep(random.uniform(1.1, 3.5))

                data_df = pd.DataFrame(data)
                data_df.drop_duplicates(inplace=True)
                data_df['CCAA'] = comunidad
                data_df.to_csv(os.path.join(raw_data_path, filename), encoding='utf-8', index=False, sep=';')

                # Espera tiempo adicional para seguir con siguiente CCAA
                time.sleep(random.uniform(2, 15))
    except Exception as e:
        send_telegram_message(f"Error en el script: {e}")
        raise
    finally:
        conn.close()

    send_telegram_message("El script de scraping de pisos ha finalizado.")

if __name__ == '__main__':
    main()
    time.sleep(random.uniform(30, 60))
