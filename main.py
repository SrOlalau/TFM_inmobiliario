import os

from data.POI import POI_OSM_downloader
from data.geolocations import add_geoloc, calculate_POI_distance
from datamunging import datamunging
from datatuning import criminalidadtuning
from machinelearning import machinelearning, machinelearning_pxm2
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

def criminalidad_tuning():
    # Determine the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    criminalidadtuning.main(script_dir)

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

    #Tuning de indices de criminalidad
    #criminalidad_tuning()

    # Después de hacer el scrpaping diario, vuelve a juntar todos los csv y ejecuta datamunging
    preprocesado()
    machine_learning()

    
