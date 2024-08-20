import os
import pandas as pd

def main(script_dir):
    output_path = os.path.join(script_dir, 'data/criminalidad/indice_criminalidad.csv')
    origin_path = os.path.join(script_dir, 'data/criminalidad/Ind_criminalidad_ESP.csv')
    df.to_csv(output_path, index=False)