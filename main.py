import os

from data.POI import POI_OSM_downloader
from data.geolocations import add_geoloc, calculate_POI_distance
from datamunging import datamunging
from data.criminalidad import indice_criminalidad
from datatuning import data_tuning
from machinelearning import machinelearning
from web_scraping_scripts import web_scraping_pisos, web_scraping_trovit
from datetime import datetime
import time

def scraping():
    # Funciones que hacen web scraping
    web_scraping_pisos.main()
    web_scraping_trovit.main()

def descarga_info_adicional():
    # Point Of Interest de Open Street Maps
    POI_OSM_downloader.main()

def preprocesado():
    # Junta todos los CSV de datos descargados
    datamunging.main()
    # Abre bbdd de datamunging y agrega geolocalizacion
    add_geoloc.main()
    calculate_POI_distance.main()
    indice_criminalidad.main()

def datatuning():
    data_tuning.main()

def machine_learning ():
    # Obtener la fecha y hora actuales
    fecha_hora_formateada = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Fecha y hora actual: {fecha_hora_formateada}')
    start_time = time.time()
    # Modelos de predicción de variable objetivo
    machinelearning.main()
    end_time = time.time()
    print(f"MachineLearning time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == '__main__':
    
    # Primero descarga puntos de interés
    #descarga_info_adicional()

    # Luego hace web scraping de los nuevos anuncios publicados en la última semana
    #scraping()

    # Después de hacer el scrpaping diario, vuelve a juntar todos los csv y ejecuta datamunging
    preprocesado()
    datatuning()
    machine_learning()

    
