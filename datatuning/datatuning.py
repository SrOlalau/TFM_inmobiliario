import os
import pandas as pd
import numpy as np

def main(script_dir):
    output_path = os.path.join(script_dir, 'datatuning/datatuning.csv')
    origin_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df = pd.read_csv(origin_path)
    
    # Lista de tipos de POIs
    poi_types = ['aerodrome','atm','bar','bicycle_rental','bus_stop','cafe','cinema',
                 'clinic','clothes','college','convenience','doctors','factory',
                 'fire_station','garden','gym','hospital','health','industrial',
                 'landfill','library','mall','marketplace','memorial','monument',
                 'museum','park','pharmacy','playground','police','post_office',
                 'prison','restaurant','school','sports_centre','station','theatre',
                 'townhall','viewpoint','waste_disposal','works']
    
    df.loc[:, poi_types] = np.nan
    
    df.to_csv(output_path, index=False)