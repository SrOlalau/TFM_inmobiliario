from web_scraping_scripts import web_scraping_pisos, web_scraping_trovit
from data.POI import POI_OSM_downloader
from datamunging import datamunging
from data.geolocations import add_geoloc

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
    geolocated_data = add_geoloc.main()
    print(geolocated_data.head())
    print(geolocated_data.info())


if __name__ == '__main__':
    # Primero descarga puntos de interés
    # descarga_info_adicional()

    # Luego hace web scraping de los nuevos anuncios publicados en la última semana
    # scraping()

    # Después de hacer el scrpaping diario, vuelve a juntar todos los csv y ejecuta datamunging
    preprocesado()


