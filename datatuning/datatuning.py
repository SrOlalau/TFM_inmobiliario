import pandas as pd
import numpy as np
import glob
import os

def main(script_dir):
    output_path = os.path.join(script_dir, 'datatuning/datatuning.csv')
    
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df = pd.read_csv(file_path)
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()