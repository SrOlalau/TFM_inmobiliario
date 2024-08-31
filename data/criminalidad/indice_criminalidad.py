import os
import pandas as pd
import re
# Open and read the file
def indice_criminalidad (script_dir):
#script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(script_dir, 'Ind_criminalidad_ESP.csv')

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Displaying the first few lines to understand the structure
    lines[:20]  # Displaying the first 20 lines

    
    # Initialize variables for parsing
    data = []
    current_region = None
    current_category = None

    for line in lines:
        # Clean up the line
        line = line.strip()

        # Skip empty lines
        if not line:
            continue
        
        # Check if the line is a region (all uppercase and no numbers)
        if line.isupper() and not any(char.isdigit() for char in line):
            current_region = line
            continue
        
        # Check if the line is a crime category or data
        if re.match(r'^\d+\.\s', line) or re.match(r'^[IVXL]+\.\s', line):
            # This line is a category or sub-category
            parts = line.split(',')
            current_category = parts[0].strip()
            values = parts[1:]
            data.append({
                'Region': current_region,
                'Category': current_category,
                'Enero-Marzo 2023': values[0].strip(),
                'Enero-Marzo 2024': values[1].strip(),
                'Porcentaje variación 2024/2023': values[2].strip()
            })
        else:
            # This is additional data for the current category
            parts = line.split(',')
            if current_category:
                data.append({
                    'Region': current_region,
                    'Category': parts[0].strip(),
                    'Enero-Marzo 2023': values[0].strip(),
                    'Enero-Marzo 2024': values[1].strip(),
                    'Porcentaje variación 2024/2023': values[2].strip()
                })

    

    # Convert the structured data to a DataFrame for easier visualization
    df = pd.DataFrame(data)
    df['Enero-Marzo 2023'] = pd.to_numeric(df['Enero-Marzo 2023'], errors='coerce')
    df['Enero-Marzo 2024'] = pd.to_numeric(df['Enero-Marzo 2024'], errors='coerce')
    df['Porcentaje variación 2024/2023'] = pd.to_numeric(df['Porcentaje variación 2024/2023'], errors='coerce')
    df.head(20)  # Displaying the first 20 ro
    output_path = os.path.join(script_dir, 'indice_crim_ESP.csv')

    # Guardar el DataFrame como un archivo CSV en la ruta especificada
    df.to_csv(output_path, index=False)

def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indice_criminalidad(script_dir)


if __name__ == "__main__":
    main()
