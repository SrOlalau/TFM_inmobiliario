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

from web_scraping_scripts.fake_user_agents_list import get_random_headers


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
print("Today date is: ", today)

# Calcula el número de páginas para iterar
def num_paginas(soup):    
    n_resultados = soup.find(class_ = 'grid__title')
    if n_resultados.text:
        n_resultados = int(re.findall(r'\d+', n_resultados.text.replace('.', ''))[0])
        n_paginas = math.ceil(n_resultados/30)
    return n_paginas if n_paginas else None


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
    if num_search:
        response = requests.get(url + str(num_search), headers=get_random_headers(), timeout=20)
    else:
        response = requests.get(url, headers=get_random_headers(), timeout=20)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup


# Hace las busquedas de ventas y alquileres en madrid
def main(ultimasemana =True):
    TIPOS = ['alquiler', 'venta']
    CCAA = [
        'madrid', 'barcelona', 'valencia', 'granada',
        'malaga', 'sevilla', 'cadiz', 'cantabria', 'vizcaya_bizkaia'
    ]

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

            data = []
            for i in range(1, n_pag + 1):
                anuncios = None
                while anuncios is None:
                    try:
                        print(f'Buscando página {i}/{n_pag}')
                        soup = hace_busqueda(num_search=i, tipo=tipo, ccaa=comunidad)
                        anuncios = soup.find(class_='grid').find_all(class_='ad-preview')
                    except AttributeError:
                        time.sleep(random.uniform(3, 7))

                for anunc in anuncios:
                    to_save = get_all_relevant(anunc)
                    if to_save:
                        data.append(to_save)
                time.sleep(random.uniform(1.1, 3.5))

            data_df = pd.DataFrame(data)
            data_df.drop_duplicates(inplace=True)
            data_df['CCAA'] = comunidad
            data_df.to_csv(os.path.join(raw_data_path, filename), encoding='utf-8', index=False, sep=';')

            # Espera tiempo adicional para seguir con siguiente CCAA
            time.sleep(random.uniform(2, 15))

if __name__ == '__main__':
    main()