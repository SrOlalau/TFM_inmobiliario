from web_scraping_scripts import web_scraping_pisos, web_scraping_trovit
from data.POI import POI_OSM_downloader

if __name__ == '__main__':
    # Primero descarga puntos de interés
    # POI_OSM_downloader.main()

    # Luego hace web scraping de los nuevos anuncios publicados en la última semana
    web_scraping_pisos.main()
    web_scraping_trovit.main()
