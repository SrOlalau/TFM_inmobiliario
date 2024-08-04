import math
import os
import random
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import sql

# Variables de conexión a la base de datos
DB_NAME = "trovit_scraping"
DB_USER = "trovit"
DB_PASSWORD = "trovit"
DB_HOST = "10.1.5.9"
DB_PORT = "5432"

# Aquí se inserta el segundo código para los User Agents y la función get_random_headers()
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

def get_random_headers():
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    return headers

# Constantes
MAX_PAGES = 100  # Número máximo de páginas que se puede iterar
RESULTS_PER_PAGE = 25  # Resultados que salen por página

# Returns the current local date
today = date.today()

# Calcula el número de páginas para iterar
def num_paginas(soup):
    results_counter = soup.find('div', class_='results-counter js-results-counter')
    if results_counter:
        total_count_text = results_counter.find('span', {'data-test': 'results'}).text
        total_count = int(total_count_text.replace('.', ''))
        n_paginas = math.ceil(total_count / RESULTS_PER_PAGE)
        return min(n_paginas, MAX_PAGES)
    return None

def extract_property_info(snippet):
    property_info = {}

    # Extract price
    price_element = snippet.find('span', class_='actual-price')
    if price_element:
        property_info["precio"] = price_element.text.strip()

    # Extract sub_description
    sub_desc_element = snippet.find('div', class_='item-title')
    if sub_desc_element:
        property_info["sub_descr"] = sub_desc_element.text.strip()
        link_element = snippet.find('a', class_='rd-link')
        if link_element:
            property_info["href"] = link_element['href'].strip()

    # Extract location
    location_element = snippet.find('span', class_='address')
    if location_element:
        property_info["ubicacion"] = location_element.text.strip()

    # Extract bedrooms, bathrooms, and area
    rooms_element = snippet.find('div', class_='item-property item-rooms')
    if rooms_element:
        property_info["habitaciones"] = rooms_element.text.strip()

    baths_element = snippet.find('div', class_='item-property item-baths')
    if baths_element:
        property_info["banios"] = baths_element.text.strip()

    area_element = snippet.find('div', class_='item-property item-size')
    if area_element:
        property_info["mt2"] = area_element.text.strip()

    # Extract additional details if available
    desc_element = snippet.find('div', class_='item-description')
    if desc_element:
        property_info["otros"] = desc_element.text.strip()

    # Extract extra info
    extra_info_element = snippet.find('div', class_='item-extra-info')
    if extra_info_element:
        published_time_element = extra_info_element.find('span', class_='item-published-time')
        if published_time_element:
            property_info["publicado_hace"] = published_time_element.text.strip()
        
        platform_element = extra_info_element.find('span', class_='item-source')
        if platform_element:
            property_info["plataforma"] = platform_element.text.strip()

    return property_info

def get_all_relevant(anunc):
    snippet = anunc.find('div', class_='snippet-content')
    to_save = extract_property_info(snippet)
    return to_save

def hace_busqueda(num_search=None, url=None):
    if num_search:
        url = url.rstrip('/') + '/' + str(num_search)
    response = requests.get(url, headers=get_random_headers(), timeout=20)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

def generate_search_url(location, type_search, page=1, subsample='Ultima semana', order='source_date'):
    # Base URL
    base_url = "https://casas.trovit.es/index.php/cod.search_adwords_homes/"

    # Type mapping
    type_mapping = {
        'venta': '1',
        'alquiler': '2'
    }

    # Order mapping
    order_mapping = {
        'source_date': 'source_date',
        'relevance': 'relevance'
    }

    # Subsample mapping
    subsample_mapping = {
        'Ultima semana': '7',
        'Ultimo dia': '1'
    }

    # Validate type
    if type_search not in type_mapping:
        raise ValueError("Invalid type. Choose 'venta' or 'alquiler'.")

    # Validate order
    if order not in order_mapping:
        raise ValueError("Invalid order. Choose 'source_date' or 'relevance'.")

    # Build URL
    url = f"{base_url}type.{type_mapping[type_search]}/what_d.{location}/order_by.{order_mapping[order]}"

    # Add subsample if provided
    if subsample:
        if subsample not in subsample_mapping:
            raise ValueError("Invalid subsample. Choose 'Ultima semana' or 'Ultimo dia'.")
        url += f"/date_from.{subsample_mapping[subsample]}"

    # Add page
    url += f"/page.{page}"

    return url

def generate_urls(provincias):
    urls = []
    for provincia in provincias:
        alquiler_url = generate_search_url(location=provincia, type_search="alquiler", subsample="Ultima semana")
        venta_url = generate_search_url(location=provincia, type_search="venta", subsample="Ultima semana")
        urls.extend([
            (alquiler_url, "alquiler", provincia),
            (venta_url, "venta", provincia)
        ])
    return urls

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
            CREATE TABLE IF NOT EXISTS scraping_trovit_tabla (
                id SERIAL PRIMARY KEY,
                precio VARCHAR(50),
                sub_descr TEXT,
                href TEXT,
                ubicacion TEXT,
                habitaciones INTEGER,
                banios INTEGER,
                mt2 INTEGER,
                otros TEXT,
                publicado_hace VARCHAR(50),
                plataforma VARCHAR(100),
                CCAA VARCHAR(100),
                fecha DATE DEFAULT CURRENT_DATE
            )
        """)
        cur.execute(create_table_query)
        conn.commit()

def insert_data_to_db(conn, data):
    with conn.cursor() as cur:
        insert_query = sql.SQL("""
            INSERT INTO scraping_trovit_tabla (
                precio, sub_descr, href, ubicacion, habitaciones, banios, mt2, otros, publicado_hace, plataforma, CCAA, fecha
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
        """)
        for row in data:
            cur.execute(insert_query, (
                row.get("precio"),
                row.get("sub_descr"),
                row.get("href"),
                row.get("ubicacion"),
                row.get("habitaciones"),
                row.get("banios"),
                row.get("mt2"),
                row.get("otros"),
                row.get("publicado_hace"),
                row.get("plataforma"),
                row.get("CCAA")
            ))
        conn.commit()

def main():
    print("Today's date is: ", today)
    provincias = ["Barcelona", "Valencia", "Cantabria", "Alicante",
                  "Madrid", "Sevilla", "Bizkaia", "Malaga", "Granada"]
    URLS = generate_urls(provincias)
    random.shuffle(URLS)

    conn = connect_db()
    create_table_if_not_exists(conn)

    for url, tipo, comunidad in URLS:
        print(f'--------- {tipo.upper()} - {comunidad} ---------')
        print(url)
        soup = hace_busqueda(url=url)
        n_pag = num_paginas(soup)

        # Generar lista de números de páginas y mezclarla
        paginas = list(range(1, n_pag + 1))
        random.shuffle(paginas)

        data = []
        for idx, i in enumerate(paginas):
            anuncios = None
            while anuncios is None:
                try:
                    print(f'Buscando página {idx + 1}/{n_pag}')
                    page_url = generate_search_url(location=comunidad, type_search=tipo,
                                                   page=i, subsample="Ultima semana")
                    soup = hace_busqueda(url=page_url)
                    anuncios = soup.find_all('div', class_='snippet-wrapper')
                except AttributeError:
                    time.sleep(random.uniform(3, 7))

            for anunc in anuncios:
                to_save = get_all_relevant(anunc)
                if to_save:
                    to_save["CCAA"] = comunidad  # Add the CCAA field
                    data.append(to_save)
            time.sleep(random.uniform(1.3, 4.3))

            # Cada 20 request, espera varios minutos promedio para evitar captcha
            if (idx + 1) % 20 == 0 and (idx + 1) != n_pag:
                time.sleep(random.uniform(560, 840))

        if data:
            insert_data_to_db(conn, data)
            print(f'Inserted {len(data)} records for {comunidad} - {tipo}')

        # Espera tiempo adicional para seguir con siguiente provincia
        time.sleep(random.uniform(560, 840))

    conn.close()

if __name__ == '__main__':
    main()
