import pandas as pd
import requests
import time
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_dir, 'geocode_cache.json')

class GeocoderCache:
    def __init__(self, cache_file=cache_path):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_coordinates(self, address):
        if address in self.cache:
            cached_result = self.cache[address]
            if isinstance(cached_result, dict) and 'error' in cached_result:
                print(f"Skipping {address} due to previous error: {cached_result['error']}")
                return None, None, False
            return cached_result[0], cached_result[1], False

        try:
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data:
                lat, lon = data[0]['lat'], data[0]['lon']
                self.cache[address] = (lat, lon)
                return lat, lon, True
            else:
                error_msg = "No results found"
                self.cache[address] = {'error': error_msg}
                print(f"Error geocoding {address}: {error_msg}")
                return None, None, True
        except requests.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            self.cache[address] = {'error': error_msg}
            print(f"Error geocoding {address}: {error_msg}")
            return None, None, True
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.cache[address] = {'error': error_msg}
            print(f"Error geocoding {address}: {error_msg}")
            return None, None, True

    def process_and_geocode(self, df, address_column='ubicacion'):
        addresses = df[address_column].dropna().unique()

        geocoded_data = {'address': [], 'latitude': [], 'longitude': [], 'error': []}

        for address in addresses:
            lat, lon, request_made = self.get_coordinates(address)
            geocoded_data['address'].append(address)
            geocoded_data['latitude'].append(lat)
            geocoded_data['longitude'].append(lon)

            error = self.cache[address].get('error') if isinstance(self.cache[address], dict) else None
            geocoded_data['error'].append(error)

            if request_made:
                time.sleep(1)  # Pause only if a new request was made

        self.save_cache()

        geocoded_df = pd.DataFrame(geocoded_data)

        result_df = df.merge(geocoded_df, left_on=address_column, right_on='address', how='left')

        return result_df


# Example usage:
if __name__ == "__main__":
    # Initialize the GeocoderCache
    geocoder = GeocoderCache()

    # Load your DataFrame
    base_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RawDataTFM")
    file_path = os.path.join(base_data_path,'downloaded_trovit_data_alquiler_Madrid_20240630.csv')
    df = pd.read_csv(file_path, sep=';')

    # Process and geocode the data
    result_df = geocoder.process_and_geocode(df, address_column='ubicacion')

    # Save the resulting DataFrame to a new CSV file
    output_file_path = r'C:\Users\alons\Documents\ARCHIVOS PERSONALES\MS DATA SCIENCE UCM\TFM\paralelo/geocoded_data.csv'
    result_df.to_csv(output_file_path, index=False)

    print(f"Geocoded data saved to {output_file_path}")
