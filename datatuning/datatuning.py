import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def main(script_dir):
    output_path = os.path.join(script_dir, 'datatuning/datatuning.csv')
    origin_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df = pd.read_csv(origin_path, low_memory=False)
    
    # Lista de tipos de POIs
    poi_types = ['aerodrome','atm','bar','bicycle_rental','bus_stop','cafe','cinema',
                 'clinic','clothes','college','convenience','doctors','factory',
                 'fire_station','garden','gym','hospital','health','industrial',
                 'landfill','library','mall','marketplace','memorial','monument',
                 'museum','park','pharmacy','playground','police','post_office',
                 'prison','restaurant','school','sports_centre','station','theatre',
                 'townhall','viewpoint','waste_disposal','works']
    
    df.loc[:, poi_types] = np.nan
    
    #funciones desde el EDA
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df =df.drop(['planta','publicado_hace'],axis=1)
    #df = df.fillna(df.median())
    #to_factor = list(df.loc[:,df.nunique() < 20])
    #df[to_factor] = df[to_factor].astype('category')
    df = df.select_dtypes(include=['int64', 'float64'])
    #____
    df.to_csv(output_path, index=False)