import math
import os
import random
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

from web_scraping_scripts.fake_user_agents_list import get_random_headers

# Constantes
MAX_PAGES = 100  # Número máximo de páginas que se puede iterar
RESULTS_PER_PAGE = 25  # Resultados que salen por página


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
print("Today's date is: ", today)


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


def generate_urls(provincias):
    base_url = "https://casas.trovit.es"
    urls = []

    # Lista de provincias bajo nombre comunidad
    comunidad = ["Madrid"]

    for provincia in provincias:
        # Determinar si debemos agregar "-provincia" al final
        suffix = "-comunidad" if provincia in comunidad else "-provincia"

        alquiler_url = f"{base_url}/alquiler-{provincia.lower().replace(' ', '-')}{suffix}"
        venta_url = f"{base_url}/{provincia.lower().replace(' ', '-')}{suffix}-piso"

        urls.extend([
            (alquiler_url, "alquiler", provincia),
            (venta_url, "venta", provincia)
        ])
    return urls



def main():
    provincias = ["Barcelona", "Valencia", "Cantabria", "Alicante",
                  "Madrid", "Sevilla", "Bizkaia", "Malaga", "Granada"]
    random.shuffle(provincias)
    URLS = generate_urls(provincias)

    for url, tipo, comunidad in URLS:
        filename = f'downloaded_trovit_data_{tipo}_{comunidad}_{today.year}{today.month:02d}{today.day:02d}.csv'
        filepath = os.path.join(raw_data_path, filename)
        if os.path.exists(filepath):
            print(f'El archivo {filename} ya existe. Pasando al siguiente archivo.')
            continue

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
                    soup = hace_busqueda(num_search=i, url=url)
                    anuncios = soup.find_all('div', class_='snippet-wrapper')
                except AttributeError:
                    time.sleep(random.uniform(3, 7))

            for anunc in anuncios:
                to_save = get_all_relevant(anunc)
                if to_save:
                    data.append(to_save)
            time.sleep(random.uniform(1.3, 4.3))

            # A la mitad de las 100 request, espera 1,25 minutos promedio para evitar captcha
            if (idx + 1) % 20 == 0:
                time.sleep(random.uniform(500, 840))

        data_df = pd.DataFrame(data)
        print(f'Downloaded {len(data_df)} obs. for {comunidad} - {tipo}')
        data_df.drop_duplicates(inplace=True)
        data_df['CCAA'] = comunidad
        data_df.to_csv(os.path.join(raw_data_path, filename), encoding='utf-8', index=False, sep=';')
        print(f'Saving {len(data_df)} obs. after drop duplicates')
        # Espera tiempo adicional para seguir con siguiente provincia
        time.sleep(random.uniform(10, 20))

if __name__ == '__main__':
    main()
