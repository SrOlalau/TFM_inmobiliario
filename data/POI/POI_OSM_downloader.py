# OSM Downloader Script
# This script downloads OpenStreetMap data for a specified location and saves it to the given path.
import requests
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_first_existing_key(tags, POI):
    for key in POI:
        if key in tags:
            return tags[key]
    return None


def osm_downloader(location):
    """
    Descarga puntos de interés de OpenStreetMap para una ubicación específica y guarda los datos en un archivo CSV.

    Parámetros:
    location (str): El nombre de la ubicación (ej. "Madrid").
    """

    # URL de la API de Overpass
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Consulta Overpass API para obtener diversos puntos de interés en la ubicación especificada
    overpass_query = f"""
    [out:json];
    area[name="{location}"]->.searchArea;
    (
      node(area.searchArea)["shop"="supermarket"];
      node(area.searchArea)["shop"="convenience"];
      node(area.searchArea)["shop"="marketplace"];
      node(area.searchArea)["amenity"="pharmacy"];
      node(area.searchArea)["amenity"="hospital"];
      node(area.searchArea)["amenity"="clinic"];
      node(area.searchArea)["amenity"="doctors"];
      node(area.searchArea)["amenity"="health"];
      node(area.searchArea)["amenity"="gym"];
      node(area.searchArea)["amenity"="school"];
      node(area.searchArea)["amenity"="college"];
      node(area.searchArea)["amenity"="library"];
      node(area.searchArea)["public_transport"="station"];
      node(area.searchArea)["highway"="bus_stop"];
      node(area.searchArea)["railway"="station"];
      node(area.searchArea)["amenity"="bicycle_rental"];
      node(area.searchArea)["leisure"="park"];
      node(area.searchArea)["leisure"="garden"];
      node(area.searchArea)["leisure"="playground"];
      node(area.searchArea)["leisure"="sports_centre"];
      node(area.searchArea)["shop"="mall"];
      node(area.searchArea)["shop"="clothes"];
      node(area.searchArea)["amenity"="cinema"];
      node(area.searchArea)["amenity"="theatre"];
      node(area.searchArea)["amenity"="restaurant"];
      node(area.searchArea)["amenity"="cafe"];
      node(area.searchArea)["amenity"="bar"];
      node(area.searchArea)["tourism"="museum"];
      node(area.searchArea)["historic"="monument"];
      node(area.searchArea)["historic"="memorial"];
      node(area.searchArea)["tourism"="viewpoint"];
      node(area.searchArea)["amenity"="post_office"];
      node(area.searchArea)["amenity"="police"];
      node(area.searchArea)["amenity"="fire_station"];
      node(area.searchArea)["amenity"="townhall"];
      node(area.searchArea)["amenity"="waste_disposal"];
      node(area.searchArea)["landuse"="landfill"];
      node(area.searchArea)["landuse"="industrial"];
      node(area.searchArea)["amenity"="prison"];
      node(area.searchArea)["man_made"="works"];
      node(area.searchArea)["amenity"="factory"];
      node(area.searchArea)["amenity"="atm"];
      node(area.searchArea)["aeroway"="aerodrome"];
    );
    out center;
    """

    # Realizar la consulta a Overpass API
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    # Main POI tag
    POI = ['amenity', 'shop', 'leisure', 'public_transport', 'tourism', 'historic', 'landuse', 'man_made', 'aeroway']

    # Procesar datos
    points_of_interest = []
    for element in data['elements']:
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
        elif 'center' in element:
            lat = element['center']['lat']
            lon = element['center']['lon']
        else:
            continue

        tags = element.get('tags', {})
        poi_type = get_first_existing_key(tags, POI) if tags else 'N/A'
        poi_name = tags.get('name', 'Unnamed')
        poi_tags = ', '.join([f"{k}: {v}" for k, v in tags.items()])

        points_of_interest.append([poi_type, poi_name, lat, lon, poi_tags])

    # Convertir a DataFrame
    df = pd.DataFrame(points_of_interest, columns=['type', 'name', 'lat', 'lon', 'tags'])
    return data, df


def main(save_dir=script_dir):
    ccaa = [
        'Madrid', 'Catalunya', 'València', 'Granada',
        'Málaga', 'Sevilla', 'Cádiz', 'Cantabria', 'Euskadi'
    ]

    # Initialize an empty list to hold the dataframes
    dfs = []

    # Iterate over the CCAA list, call OSM_downloader, and append the dataframe to the list
    for region in ccaa:
        print(f'Realizando consulta a OSM, via Overpass API: {region}')
        data, df = osm_downloader(region)
        dfs.append(df)

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)

    save_path = os.path.join(save_dir, 'points_of_interest_ES.csv')
    # Save the combined dataframe to a CSV file
    combined_df.to_csv(save_path, index=False)
    print(f'Data saved on: {save_path}')


if __name__ == '__main__':
    main()