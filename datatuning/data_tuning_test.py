import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class GroupedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_columns, group_column, strategy='most_frequent'):
        self.cat_columns = cat_columns
        self.group_column = group_column
        self.strategy = strategy
        self.imputers = {col: SimpleImputer(strategy=self.strategy) for col in self.cat_columns}

    def fit(self, X, y=None):
        # Ajustar los imputadores a los datos de cada grupo
        grouped = X.groupby(self.group_column)
        for col in self.cat_columns:
            for _, group in grouped:
                self.imputers[col].fit(group[[col]])
        return self

    def transform(self, X):
        print(f'imputando:{self.cat_columns}')
        # Aplicar la imputación a cada grupo
        X_imputed = X.groupby(self.group_column, group_keys=False).apply(self._impute_group)
        return X_imputed

    def _impute_group(self, group):
        for col in self.cat_columns:
            group[col] = self.imputers[col].transform(group[[col]]).ravel()
        return group

def data_tuning(script_dir):
    # Ruta del archivo CSV
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data_DM.csv')
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(file_path)
    #Eliminar filas donde 'precio' es NaN, 0, o infinito
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df.drop(['planta', 'publicado_hace', 'raw_json', 'fuente_datos','sub_descr', 'href', 'ubicacion', 'otros'], axis=1)
    print("variables categoricas del DF son:")
    print(df.select_dtypes(include=['category','object']).columns.tolist())
    print("variables numericas del DF son:")
    print(df.select_dtypes(include=['number']).columns.tolist())
    #CAMBIO INCORPORADO DESDE EL EDA
    # Se tramifica variable numérica y se vuelve de tipo objeto
#    df['habitaciones'] = df['habitaciones'].apply(lambda x: '8 o más' if x >= 8 else x)
#    df['banios'] = df['banios'].apply(lambda x: '7 o más' if x >= 7 else x)
    #CAMBIO INCORPORADO DESDE EL EDA
    #Homegeneizamos la comunidad autónoma
    df['CCAA'] = df['CCAA'].replace({
    'Malaga': 'malaga',
    'Sevilla': 'sevilla',
    'Granada': 'granada',
    'Madrid': 'madrid',
    'Barcelona': 'barcelona',
    'vizcaya_bizkaia': 'bizkaia',
    'Bizkaia': 'bizkaia',
    'Cantabria': 'cantabria',
    'Valencia': 'valencia',
    'Alicante': 'alicante'
    })
    #Reemplazar missings no declarados (ceros) por Nan
    num_var = ['mt2', 'latitude', 'longitude']
    for col in num_var:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)
    #Modificar tipo object por category, variables de menos de 20 valores unicos
    to_factor = list(df.loc[:,df.nunique() < 20])
    df[to_factor] = df[to_factor].astype('category')
    #Sustituir Nan en plataforma por Pisos.com ya que era la única sin esa variable
    df['plataforma']= df['plataforma'].replace(np.nan, 'Pisos')
    
    #Imputaciones categóricas por moda y numericas por promedio
    cat_cols = df.select_dtypes(include=['category','object']).columns.tolist()
    cat_imputer = GroupedImputer(cat_cols,'CCAA')
    df = cat_imputer.fit_transform(df)

    num_cols = ['habitaciones', 'banios', 'mt2']
    num_imputer = GroupedImputer(num_cols,'CCAA','mean')
    df = num_imputer.fit_transform(df)

    #def winsorize_with_pandas(s, limits):
    #    """
    #    s : pd.Series
    #        Series to winsorize
    #    limits : tuple of float
    #        Tuple of the percentages to cut on each side of the array, 
    #        with respect to the number of unmasked data, as floats between 0. and 1
    #    """
    #    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
    #                upper=s.quantile(1-limits[1], interpolation='higher'))
    
    #def gestiona_outliers(col, clas='check'):
    #    """
    #    Función para gestionar outliers en una Serie.
    #    """
    #    # Condición de asimetría y aplicación de criterio 1 según el caso
    #    if abs(col.skew()) < 1:
    #        criterio1 = abs((col - col.mean()) / col.std()) > 3
    #    else:
    #        criterio1 = abs((col - col.median()) / stats.median_abs_deviation(col)) > 8
    #    
    #    # Calcular primer cuartil     
    #    q1 = col.quantile(0.25)  
    #    # Calcular tercer cuartil  
    #    q3 = col.quantile(0.75)
    #    # Calculo de IQR
    #    IQR = q3 - q1
    #    # Calcular criterio 2 (general para cualquier asimetría)
    #    criterio2 = (col < (q1 - 3 * IQR)) | (col > (q3 + 3 * IQR))
    #    lower = col[criterio1 & criterio2 & (col < q1)].count() / col.dropna().count()
    #    upper = col[criterio1 & criterio2 & (col > q3)].count() / col.dropna().count()
    #    # Salida según el tipo deseado
    #    if clas == 'winsor':
    #        return winsorize_with_pandas(col, (lower, upper))
    
    # Aplicar la función de outliers a la columna completa
    #df['mt2'] = gestiona_outliers(df['mt2'], clas='winsor')

    output_path = os.path.join(script_dir, 'datamunging/consolidated_data_DT.csv')
    df.to_csv(output_path, index=False)


def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_tuning(script_dir)

if __name__ == "__main__":
    main()