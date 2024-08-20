from web_scraping_scripts import web_scraping_pisos, web_scraping_trovit
from data.POI import POI_OSM_downloader
from datamunging import datamunging
from machinelearning import machinelearning
from data.geolocations import add_geoloc
from datatuning import datatuning
from datatuning import criminalidadtuning
import os

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

def criminalidad_tuning():
    # Determine the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    criminalidadtuning.main(script_dir)

def data_tuning():
    # Determine the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Mejora de variables y preprocesado de información
    datatuning.main(script_dir)

def machine_learning ():
    # Modelos de predicción de variable objetivo
    machinelearning.main()

if __name__ == '__main__':
    # Primero descarga puntos de interés
    #descarga_info_adicional()

    # Luego hace web scraping de los nuevos anuncios publicados en la última semana
    #scraping()

    #Tuning de indices de criminalidad
    criminalidad_tuning()

    # Después de hacer el scrpaping diario, vuelve a juntar todos los csv y ejecuta datamunging
    preprocesado()
    data_tuning()
    machine_learning()


