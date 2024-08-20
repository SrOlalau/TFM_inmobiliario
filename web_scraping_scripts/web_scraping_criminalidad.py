import os

def main(script_dir):
    output_path = os.path.join(script_dir, 'data/criminalidad/indice_criminalidad.csv')
    
    df.to_csv(output_path, index=False)