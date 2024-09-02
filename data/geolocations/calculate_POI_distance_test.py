# Script temporal para agregar la función que calcula distancias.

import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class POI_counter:
    def __init__(self, poi_path):
        self.poi_df = pd.read_csv(poi_path)
        self.filtered_poi_df = self._filter_poi_types()
        self.poi_types = self.filtered_poi_df['type'].dropna().unique()
        self.selected_types = self.poi_types
        self.radii = [1]  # Default radius
        self.df_coords = None
        self.na_mask = None
        self.closest_types = []  # New attribute for closest POI calculation

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
        filtered_pois = POI_counter._bounding_box_filter(center, pois, radius)
        if len(filtered_pois) == 0:
            return 0
        squared_distances = POI_counter._squared_euclidean_distance(center[0], center[1], filtered_pois[:, 0],
                                                                    filtered_pois[:, 1])
        return np.sum(squared_distances <= radius ** 2)

    @staticmethod
    def _count_pois_for_multiple_centers(pois, centers, radius):
        return Parallel(n_jobs=-1)(
            delayed(POI_counter._count_pois_within_radius)(center, pois, radius) for center in centers)

    def set_closest_types(self, types):
        if types == 'all':
            self.closest_types = self.selected_types
        else:
            if not set(types).issubset(self.poi_types):
                raise ValueError("Some of the provided types are not valid.")
            self.closest_types = types

    @staticmethod
    def _find_closest_poi(center, pois):
        if len(pois) == 0:
            return np.inf
        distances = POI_counter._squared_euclidean_distance(center[0], center[1], pois[:, 0], pois[:, 1])
        return np.sqrt(np.min(distances))

    def _find_closest_pois_for_multiple_centers(self, pois, centers):
        return Parallel(n_jobs=-1)(delayed(self._find_closest_poi)(center, pois) for center in centers)

    def get_poi_types(self):
        type_counts = Counter(self.filtered_poi_df['type'])
        return dict(type_counts)

    def set_poi_types(self, types):
        if not set(types).issubset(self.poi_types):
            raise ValueError("Some of the provided types are not valid.")
        self.selected_types = types

    def set_radii(self, radii):
        self.radii = radii

    def calculate_poi_counts(self, df):
        result_df = df.copy()

        if self.df_coords is None:
            self.na_mask = result_df[['latitude', 'longitude']].isna().any(axis=1)
            self.df_coords = result_df.dropna(subset=['latitude', 'longitude'])[['latitude', 'longitude']].to_numpy()

        start_time = time.time()

        new_columns = {}

        # Calculate POI counts
        for radius in self.radii:
            for poi_type in self.selected_types:
                print(f'Calculating for {poi_type} at {radius}km radius')
                poi_type_df = self.filtered_poi_df[self.filtered_poi_df['type'] == poi_type]
                pois = poi_type_df[['lat', 'lon']].to_numpy()

                poi_counts = self._count_pois_for_multiple_centers(pois, self.df_coords, radius)

                col_name = f'POI_{poi_type}_{radius}km'
                new_columns[col_name] = pd.Series(poi_counts, index=result_df.index[~self.na_mask])

        # Calculate distance to closest POI
        for poi_type in self.closest_types:
            print(f'Calculating closest distance for {poi_type}')
            poi_type_df = self.filtered_poi_df[self.filtered_poi_df['type'] == poi_type]
            pois = poi_type_df[['lat', 'lon']].to_numpy()

            # Use a 5km bounding box for initial filtering
            filtered_pois = [self._bounding_box_filter(center, pois, 5) for center in self.df_coords]
            closest_distances = self._find_closest_pois_for_multiple_centers(pois, self.df_coords)

            col_name = f'closest_{poi_type}'
            new_columns[col_name] = pd.Series(closest_distances, index=result_df.index[~self.na_mask])

        # Convert new_columns dictionary to DataFrame and join to result_df
        new_columns_df = pd.DataFrame(new_columns)
        result_df = pd.concat([result_df, new_columns_df], axis=1)

        # Handle NaN values
        for col in new_columns_df.columns:
            result_df.loc[self.na_mask, col] = np.nan

        end_time = time.time()
        print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")

        return result_df


def main(output_file_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)),
                                    'datamunging/consolidated_data_DM.csv')

    POI_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)),
                                    'data/POI/points_of_interest_ES.csv')

    # Initialize the class
    poi_counter = POI_counter(POI_path)

    # See a list of the types (and the points frequency)
    poi_types = poi_counter.get_poi_types()
    print("Available POI types and their frequencies:")
    for poi_type, count in poi_types.items():
        print(f"{poi_type}: {count}")
    print('## CALCULATING DISTANCES ##')

    # Select the types you want to setup for the calculus
    selected_types = ['aerodrome', 'restaurant']  # Replace with actual types
    # poi_counter.set_poi_types(selected_types) # Comentado para que realice el cálculo para todos los tipos
    radii = [1, 3, 5]
    poi_counter.set_radii(radii)

    # Set up closest POI calculation
    closest_types = ['aerodrome', 'mall', 'cinema', 'hospital', 'park', ]
    poi_counter.set_closest_types('all')

    df = pd.read_csv(data_path, low_memory=False)

    # Calculate the points
    result_df = poi_counter.calculate_poi_counts(df)

    # Save the resulting DataFrame to a new CSV file
    result_df.to_csv(data_path, index=False)
    print(f"Data with POI saved to {data_path}")


if __name__ == "__main__":
    main()
