import os

from data.POI import POI_OSM_downloader
from data.geolocations import add_geoloc, calculate_POI_distance
from datamunging import datamunging
from data.criminalidad import indice_criminalidad
from datatuning import data_tuning
from machinelearning import machine_learning
from web_scraping_scripts import web_scraping_pisos, web_scraping_trovit
from EDA import EDA_graph_table_gen

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
    #indice_criminalidad.main()

def datatuning():
    data_tuning.main()

def EDA():
    EDA_graph_table_gen.main()

def ML():
    # Modelos de predicción de variable objetivo
    machine_learning.main()

if __name__ == '__main__':
    
    # Primero descarga puntos de interés
    #descarga_info_adicional()

    # Luego hace web scraping de los nuevos anuncios publicados en la última semana
    #scraping()

    # Después de hacer el scrpaping diario, vuelve a juntar todos los csv y ejecuta datamunging
    preprocesado()
    datatuning()
    EDA()
    ML()

    
