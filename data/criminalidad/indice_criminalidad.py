import os
import pandas as pd
import re

# Open and read the file
def indice_criminalidad(script_dir):
    file_path = os.path.join(script_dir, 'criminalidad/Ind_criminalidad_ESP.csv')

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize variables for parsing
    data = []
    current_region = None
    current_category = None
    current_postal_code = None

    for line in lines:
        # Clean up the line
        line = line.strip()

        # Skip empty lines
        if not line:
            continue
        
        # Check if the line is a region (all uppercase and no numbers)
        if line.isupper() and not any(char.isdigit() for char in line):
            # Remove commas and quotes from the region name
            current_region = line.replace(',', '').replace('"', '').strip()
            continue

        # Check if the line contains a postal code
        postal_code_match = re.match(r'^(\d{5})\s+(.+)', line)
        if postal_code_match:
            current_postal_code = postal_code_match.group(1)
            current_category = postal_code_match.group(2).strip()
            continue
        
        # Check if the line is a crime category or data
        if re.match(r'^\d+\.\s', line) or re.match(r'^[IVXL]+\.\s', line):
            # Remove numbering from the category
            category_without_number = re.sub(r'^\d+\.\s|^[IVXL]+\.\s|\d+\.\d+\.\-\s*', '', line)
            parts = category_without_number.split(',')
            current_category = parts[0].strip()
            values = parts[1:]
            data.append({
                'Region': current_region,
                'Postal Code': current_postal_code,
                'Category': current_category,
                'Enero-Marzo 2023': values[0].strip() if len(values) > 0 else None,
                'Enero-Marzo 2024': values[1].strip() if len(values) > 1 else None,
                'Porcentaje variación 2024/2023': values[2].strip() if len(values) > 2 else None
            })
        else:
            # This is additional data for the current category
            parts = line.split(',')
            if current_category:
                data.append({
                    'Region': current_region,
                    'Postal Code': current_postal_code,
                    'Category': re.sub(r'^\d+\.\d+\.\-\s*', '', parts[0].strip()),  # Clean the category field
                    'Enero-Marzo 2023': parts[1].strip() if len(parts) > 1 else None,
                    'Enero-Marzo 2024': parts[2].strip() if len(parts) > 2 else None,
                    'Porcentaje variación 2024/2023': parts[3].strip() if len(parts) > 3 else None
                })

    # Convert the structured data to a DataFrame for easier visualization
    df = pd.DataFrame(data)

    # Fill down the postal codes to ensure all rows have the correct postal code
    df['Postal Code'] = df['Postal Code'].fillna(method='ffill')

    # Convert numeric columns
    df['Enero-Marzo 2023'] = pd.to_numeric(df['Enero-Marzo 2023'], errors='coerce')
    df['Enero-Marzo 2024'] = pd.to_numeric(df['Enero-Marzo 2024'], errors='coerce')
    df['Porcentaje variación 2024/2023'] = pd.to_numeric(df['Porcentaje variación 2024/2023'], errors='coerce')

    # Save the DataFrame to a CSV file
    output_path = os.path.join(script_dir, 'criminalidad/indice_crim_ESP.csv')
    df.to_csv(output_path, index=False)

    # Display the first few rows
    print(df.head(20))

def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indice_criminalidad(script_dir)

if __name__ == "__main__":
    main()