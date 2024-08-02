import pandas as pd
import requests
import time
import json
import os

class GeocoderCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Cache file {self.cache_file} is corrupted. Starting with an empty cache.")
                return {}
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
        # Filter for rows where latitude or longitude is NA
        mask = df['latitude'].isna() | df['longitude'].isna()
        addresses_to_geocode = df.loc[mask, address_column].dropna().unique()

        for address in addresses_to_geocode:
            lat, lon, request_made = self.get_coordinates(address)

            if lat is not None and lon is not None:
                # Update the DataFrame in-place
                df.loc[(df[address_column] == address) & mask, 'latitude'] = float(lat)
                df.loc[(df[address_column] == address) & mask, 'longitude'] = float(lon)
            else:
                error = self.cache[address].get('error') if isinstance(self.cache[address], dict) else None
                df.loc[(df[address_column] == address) & mask, 'geocoding_error'] = error

            if request_made:
                time.sleep(1)  # Pause only if a new request was made

        self.save_cache()
        return df


def main(output_file_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, 'geocode_cache.json')
    # Initialize the GeocoderCache
    geocoder = GeocoderCache(cache_file=cache_path)

    datamunging_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)),
                                    'datamunging/consolidated_data.csv')
    df = pd.read_csv(datamunging_path, low_memory=False)

    # Process and geocode the data
    result_df = geocoder.process_and_geocode(df, address_column='ubicacion')

    # Si está definido el output_save_path, se guardan los datos como csv, si no, entonces hace un return de los datos
    if output_file_path:
        # Save the resulting DataFrame to a new CSV file
        result_df.to_csv(output_file_path, index=False)
        print(f"Geocoded data saved to {output_file_path}")
        return None

    return result_df

# Example usage:
if __name__ == "__main__":
    main()
