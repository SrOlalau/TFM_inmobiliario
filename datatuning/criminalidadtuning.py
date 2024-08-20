import os
import pandas as pd
import re

def main(script_dir):
    output_path = os.path.join(script_dir, 'datatuning/indice_criminalidad.csv')
    origin_path = os.path.join(script_dir, 'data/criminalidad/Ind_criminalidad_ESP.csv')
    data = []
    region = None

    # Mapeo de caracteres incorrectos a correctos
    replacements = {
        "Ã": "I",  # A veces puede ser "I"
        "Ã¡": "a",
        "Ã©": "e",
        "Ã­": "i",
        "Ã³": "o",
        "Ãº": "u",
        "Ã±": "ñ",
        "Â": "",   # Puede eliminarse si es un carácter incorrecto adicional
        "©": "e",
        "º": "o"
    }

    with open(origin_path, 'r', encoding='latin1') as file:
        for i, line in enumerate(file):
            if i < 5:  # Saltar las primeras 5 líneas de metadatos
                continue
            columns = line.split(',')
            
            # Si la línea tiene una sola columna, asumimos que es un nombre de región
            if len(columns) == 2:
                region = columns[0].strip()
                # Eliminar los números y otros caracteres no deseados
                region = re.sub(r'\d+', '', region)
                region = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', region).strip()

                # Reemplazar caracteres incorrectos con los correctos
                for wrong_char, correct_char in replacements.items():
                    region = region.replace(wrong_char, correct_char)
                
                region = region.strip()
                continue
            
            # Añadir el nombre de la región como primera columna
            if region:
                columns.insert(0, region)
                data.append(columns)
    
    # Crear el DataFrame sin limitar el número de columnas
    df = pd.DataFrame(data)
    
    # Verificar las regiones detectadas
    unique_regions = df[0].unique()
    print("Regiones detectadas:", unique_regions)
    #def unique
    unique_regions = df[0].unique()
    # Mapeo de correcciones específicas para los nombres de regiones ya detectados
    corrections = {
        "ANDALUCIA": "Andalucía",
        "Provincia de ALMERIA": "Provincia de Almería",
        "Adra": "Adra",
        "AlmerIa": "Almería",
        "Ejido (El)": "El Ejido",
        "HuIercal-Overa": "Huércal-Overa",
        "NIjar": "Níjar",
        "Roquetas de Mar": "Roquetas de Mar",
        "VIcar": "Vícar",
        "Provincia de CIDIZ": "Provincia de Cádiz",
        "Algeciras": "Algeciras",
        "Arcos de la Frontera": "Arcos de la Frontera",
        "Barbate": "Barbate",
        "Barrios (Los)": "Los Barrios",
        "CI¡diz": "Cádiz",
        "Chiclana de la Frontera": "Chiclana de la Frontera",
        "Conil de la Frontera": "Conil de la Frontera",
        "Jerez de la Frontera": "Jerez de la Frontera",
        "LInea de la ConcepciI³n (La)": "Línea de la Concepción",
        "Puerto de Santa MarIa (El)": "Puerto de Santa María",
        "Puerto Real": "Puerto Real",
        "Rota": "Rota",
        "San Fernando": "San Fernando",
        "San Roque": "San Roque",
        "SanlIocar de Barrameda": "Sanlúcar de Barrameda",
        "Provincia de CIRDOBA": "Provincia de Córdoba",
        "Cabra": "Cabra",
        "CI³rdoba": "Córdoba",
        "Lucena": "Lucena",
        "Montilla": "Montilla",
        "Palma del RIo": "Palma del Río",
        "Priego de CI³rdoba": "Priego de Córdoba",
        "Puente Genil": "Puente Genil",
        "Provincia de GRANADA": "Provincia de Granada",
        "AlmuI±Iecar": "Almuñécar",
        "Armilla": "Armilla",
        "Atarfe": "Atarfe",
        "Baza": "Baza",
        "Gabias (Las)": "Las Gabias",
        "Granada": "Granada",
        "Loja": "Loja",
        "Maracena": "Maracena",
        "Motril": "Motril",
        "Provincia de HUELVA": "Provincia de Huelva",
        "Aljaraque": "Aljaraque",
        "Almonte": "Almonte",
        "Ayamonte": "Ayamonte",
        "Cartaya": "Cartaya",
        "Huelva": "Huelva",
        "Isla Cristina": "Isla Cristina",
        "Lepe": "Lepe",
        "Moguer": "Moguer",
        "Provincia de JAIN": "Provincia de Jaén",
        "AlcalI¡ la Real": "Alcalá la Real",
        "AndIojar": "Andújar",
        "JaIen": "Jaén",
        "Linares": "Linares",
        "Martos": "Martos",
        "Ibeda": "Úbeda",
        "Provincia de MILAGA": "Provincia de Málaga",
        "AlhaurIn de la Torre": "Alhaurín de la Torre",
        "AlhaurIn el Grande": "Alhaurín el Grande",
        "Antequera": "Antequera",
        "BenalmI¡dena": "Benalmádena",
        "CI¡rtama": "Cártama",
        "CoIn": "Coín",
        "Estepona": "Estepona",
        "Fuengirola": "Fuengirola",
        "MI¡laga": "Málaga",
        "Marbella": "Marbella",
        "Mijas": "Mijas",
        "Nerja": "Nerja",
        "RincI³n de la Victoria": "Rincón de la Victoria",
        "Ronda": "Ronda",
        "Torremolinos": "Torremolinos",
        "Torrox": "Torrox",
        "VIelez-MI¡laga": "Vélez-Málaga",
        "Provincia de SEVILLA": "Provincia de Sevilla",
        "AlcalI¡ de GuadaIra": "Alcalá de Guadaíra",
        "Bormujos": "Bormujos",
        "Camas": "Camas",
        "Carmona": "Carmona",
        "Coria del RIo": "Coria del Río",
        "Dos Hermanas": "Dos Hermanas",
        "Icija": "Écija",
        "Lebrija": "Lebrija",
        "Mairena del Alcor": "Mairena del Alcor",
        "Mairena del Aljarafe": "Mairena del Aljarafe",
        "MorI³n de la Frontera": "Morón de la Frontera",
        "Palacios y Villafranca (Los)": "Los Palacios y Villafranca",
        "Rinconada (La)": "La Rinconada",
        "San Juan de Aznalfarache": "San Juan de Aznalfarache",
        "Sevilla": "Sevilla",
        "Tomares": "Tomares",
        "Utrera": "Utrera",
        "ARAGIN": "Aragón",
        "Provincia de HUESCA": "Provincia de Huesca",
        "Huesca": "Huesca",
        "Provincia de TERUEL": "Provincia de Teruel",
        "Teruel": "Teruel",
        "Provincia de ZARAGOZA": "Provincia de Zaragoza",
        "Zaragoza": "Zaragoza",
        "ASTURIAS (PRINCIPADO DE)": "Asturias (Principado de)",
        "AvilIes": "Avilés",
        "CastrillI³n": "Castrillón",
        "GijI³n": "Gijón",
        "Langreo": "Langreo",
        "Mieres": "Mieres",
        "Oviedo": "Oviedo",
        "Siero": "Siero",
        "BALEARS (ILLES)": "Baleares (Islas)",
        "Isla de Formentera": "Isla de Formentera",
        "Isla  de Eivissa": "Isla de Ibiza",
        "Eivissa": "Ibiza",
        "Sant Antoni de Portmany": "Sant Antoni de Portmany",
        "Sant Josep de sa Talaia": "Sant Josep de sa Talaia",
        "Santa EulI\xa0ria des Riu": "Santa Eulària des Riu",
        "Isla de Mallorca": "Isla de Mallorca",
        "Alcudia": "Alcúdia",
        "CalviI": "Calvià",
        "Inca": "Inca",
        "Llucmajor": "Llucmajor",
        "Manacor": "Manacor",
        "MarratxI": "Marratxí",
        "Palma": "Palma",
        "Isla de Menorca": "Isla de Menorca",
        "Ciutadella de Menorca": "Ciutadella de Menorca",
        "MaI³-MahI³n": "Mahón",
        "CANARIAS": "Canarias",
        "Provincia de PALMAS (LAS)": "Provincia de Las Palmas",
        "Isla de Fuerteventura": "Isla de Fuerteventura",
        "Oliva (La)": "La Oliva",
        "PI¡jara": "Pájara",
        "Puerto del Rosario": "Puerto del Rosario",
        "Isla de Gran Canaria": "Isla de Gran Canaria",
        "AgI¼imes": "Agüimes",
        "Arucas": "Arucas",
        "GI¡ldar": "Gáldar",
        "Ingenio": "Ingenio",
        "MogI¡n": "Mogán",
        "Palmas de Gran Canaria (Las)": "Las Palmas de Gran Canaria",
        "San BartolomIe de Tirajana": "San Bartolomé de Tirajana",
        "Santa LucIa de Tirajana": "Santa Lucía de Tirajana",
        "Telde": "Telde",
        "Isla de Lanzarote": "Isla de Lanzarote",
        "Arrecife": "Arrecife",
        "Teguise": "Teguise",
        "TIas": "Tías",
        "Provincia de SANTA CRUZ DE TENERIFE": "Provincia de Santa Cruz de Tenerife",
        "Isla de Gomera (La)": "La Gomera",
        "Isla de Hierro (El)": "El Hierro",
        "Isla de Palma (La)": "La Palma",
        "Llanos de Aridane (Los)": "Los Llanos de Aridane",
        "Isla de Tenerife": "Isla de Tenerife"
        
        # Agrega aquí más correcciones según sea necesario
    }

    # Aplicar las correcciones a unique_regions
    corrected_regions = [corrections.get(region, region) for region in unique_regions]

    # Mostrar las regiones corregidas
    print("Regiones corregidas:", corrected_regions)

    # Reemplazar las regiones en el DataFrame original
    df[0] = df[0].replace(corrections)

    #elimino las columnas creadas.
    df = df.drop(columns=[5, 6])
    # Renombrar las columnas
    df.columns = ["Region", "Descripcion", "Periodo_2023", "Periodo_2024", "Porcentaje_variacion"]
    df.to_csv(output_path, index=False)
    print("Hola mundo")