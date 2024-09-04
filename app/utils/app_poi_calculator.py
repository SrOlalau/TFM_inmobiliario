import numpy as np
import pandas as pd
from collections import Counter


class POICalculator:
    def __init__(self, poi_path):
        # Cargar y filtrar POIs
        self.poi_df = pd.read_csv(poi_path)
        self.filtered_poi_df = self._filter_poi_types()
        self.poi_types = self.filtered_poi_df['type'].dropna().unique()

    def _filter_poi_types(self, min_count=20):
        type_counter = Counter(self.poi_df['type'])
        return self.poi_df[self.poi_df['type'].map(type_counter) >= min_count]

    @staticmethod
    def _squared_euclidean_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        x = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
        y = lat2 - lat1
        return (R * (x ** 2 + y ** 2) * np.pi / 180) ** 2

    @staticmethod
    def _bounding_box_filter(center, pois, radius):
        lat, lon = center
        delta = radius / 111

        lat_min, lat_max = lat - delta, lat + delta
        lon_min, lon_max = lon - delta, lon + delta

        return pois[(pois[:, 0] >= lat_min) & (pois[:, 0] <= lat_max) &
                    (pois[:, 1] >= lon_min) & (pois[:, 1] <= lon_max)]

    @staticmethod
    def _count_pois_within_radius(center, pois, radius):
        filtered_pois = POICalculator._bounding_box_filter(center, pois, radius)
        if len(filtered_pois) == 0:
            return 0
        squared_distances = POICalculator._squared_euclidean_distance(center[0], center[1], filtered_pois[:, 0],
                                                                      filtered_pois[:, 1])
        return np.sum(squared_distances <= radius ** 2)

    @staticmethod
    def _find_closest_poi(center, pois):
        if len(pois) == 0:
            return np.inf
        distances = POICalculator._squared_euclidean_distance(center[0], center[1], pois[:, 0], pois[:, 1])
        return np.sqrt(np.min(distances))

    def calculate_point_estimations(self, lat, lon, variables):
        """
        Calcula las métricas solicitadas para un punto dado (lat, lon).

        :param lat: Latitud del punto.
        :param lon: Longitud del punto.
        :param variables: Lista de variables a calcular en el formato ['POI_type_radius', 'closest_POI_type'].
        :return: Diccionario con resultados de las métricas calculadas.
        """
        results = {}
        center = (lat, lon)

        # Separar variables en contadores y distancias más cercanas
        poi_counts = [var for var in variables if 'POI_' in var]
        closest_pois = [var for var in variables if 'closest_' in var]

        # Calcular POI counts dentro del radio
        for count_var in poi_counts:
            parts = count_var.split('_')
            poi_type = parts[1]
            radius = float(parts[2].replace('km', ''))

            if poi_type not in self.poi_types:
                raise ValueError(f"POI type '{poi_type}' not found in available POIs.")

            poi_type_df = self.filtered_poi_df[self.filtered_poi_df['type'] == poi_type]
            pois = poi_type_df[['lat', 'lon']].to_numpy()

            count = self._count_pois_within_radius(center, pois, radius)
            results[count_var] = count

        # Calcular distancia al POI más cercano
        for closest_var in closest_pois:
            poi_type = closest_var.replace('closest_', '')

            if poi_type not in self.poi_types:
                raise ValueError(f"POI type '{poi_type}' not found in available POIs.")

            poi_type_df = self.filtered_poi_df[self.filtered_poi_df['type'] == poi_type]
            pois = poi_type_df[['lat', 'lon']].to_numpy()

            closest_distance = self._find_closest_poi(center, pois)
            results[closest_var] = closest_distance

        return results


def main():
    # Ruta del archivo de POIs
    POI_path = './data/POI/points_of_interest_ES.csv'

    # Inicializar el calculador de POIs
    poi_calculator = POICalculator(POI_path)

    # Ejemplo de uso: Calcular métricas para un punto específico
    latitude = 41.397894
    longitude = 2.143347
    variables_to_calculate = ['POI_pharmacy_5km', 'POI_school_5km', 'closest_bus_station']

    # Calcular estimaciones
    results = poi_calculator.calculate_point_estimations(latitude, longitude, variables_to_calculate)

    # Imprimir resultados
    print(results)


if __name__ == "__main__":
    main()
