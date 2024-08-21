import pandas as pd
import requests
import time
import json
import os
import re
import pgeocode
from fuzzywuzzy import fuzz

class GeocoderCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.cache_keys = set(self.cache.keys())

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
        if address in self.cache_keys:
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

    def filter_and_geocode_errors(self, error_addresses, df, address_column='ubicacion'):
        # Regular expression to match 5 digits followed by a comma
        postal_code_regex = re.compile(r'^(\d{5}),')

        # List to hold postal codes
        postal_codes = []
        address_map = {}

        # Filter and extract postal codes
        for address in error_addresses:
            if address and isinstance(address, str):
                match = postal_code_regex.match(address)
                if match:
                    postal_code = match.group(1)
                    postal_codes.append(postal_code)
                    address_map[postal_code] = address
            else:
                print(f"Skipping invalid address: {address}")

        # Query pgeocode for the postal codes
        nomi = pgeocode.Nominatim('es')
        results = nomi.query_postal_code(postal_codes)

        # Update the cache and DataFrame with the results
        for i, result in results.iterrows():
            if not pd.isna(result['latitude']) and not pd.isna(result['longitude']):
                postal_code = result['postal_code']
                lat, lon = result['latitude'], result['longitude']
                original_address = address_map[postal_code]

                # Extract relevant fields for fuzzy matching
                relevant_fields = [
                    result['place_name'],
                    result['state_name'],
                    result['community_name'],
                    result['county_name']
                ]

                # Extract the text part of the original address (after postal code)
                address_text = postal_code_regex.sub('', original_address).strip().lower()

                # Perform fuzzy matching
                valid_match = False
                for field in relevant_fields:
                    if pd.notna(field):  # Ensure the field is not NaN
                        field = field.lower()
                        score = fuzz.token_set_ratio(field, address_text)
                        if score >= 97:
                            valid_match = True
                            break

                # If a valid match is found, update cache and DataFrame
                if valid_match:
                    self.cache[original_address] = (lat, lon)
                    df.loc[(df[address_column] == original_address), 'latitude'] = float(lat)
                    df.loc[(df[address_column] == original_address), 'longitude'] = float(lon)
                else:
                    print(f"No valid match found for {original_address} with postal code {postal_code}. Skipping.")

        # Save cache after updates
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

    # Identify addresses with errors in geocoding (e.g., null lat/lon, or geocoding_error)
    error_mask = result_df['latitude'].isna() | result_df['longitude'].isna() | result_df['geocoding_error'].notna()
    error_addresses = result_df.loc[error_mask, 'ubicacion'].unique()

    # Filter and geocode these error addresses using the pgeocode library
    if len(error_addresses) > 0:
        result_df = geocoder.filter_and_geocode_errors(error_addresses, result_df, address_column='ubicacion')

    # Si está definido el output_save_path, se guardan los datos como csv, si no, entonces hace un return de los datos
    if output_file_path:
        # Save the resulting DataFrame to a new CSV file
        result_df.to_csv(output_file_path, index=False)
        print(f"Geocoded data saved to {output_file_path}")
    else:
        # Save the resulting DataFrame on the same CSV file
        result_df.to_csv(datamunging_path, index=False)
        print(f"Geocoded data saved to {datamunging_path}")

# Example usage:
if __name__ == "__main__":
    main()
