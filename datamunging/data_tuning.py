import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
from scipy import stats

def data_tuning(script_dir):
    # Ruta del archivo CSV
    file_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(file_path)
    #Eliminar filas donde 'precio' es NaN, 0, o infinito
    df = df[~df['precio'].isin([0, np.inf, -np.inf]) & df['precio'].notna()]
    df = df.drop(['planta', 'publicado_hace', 'raw_json'], axis=1)
    #CAMBIO INCORPORADO DESDE EL EDA
    # Se tramifica variable numérica y se vuelve de tipo objeto
    df['habitaciones'] = df['habitaciones'].apply(lambda x: '8 o más' if x >= 8 else x)
    df['banios'] = df['banios'].apply(lambda x: '7 o más' if x >= 7 else x)
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


    cat_cols = df.select_dtypes(include=['category','object']).columns
    #Nombre de la columna que contiene la comunidad autónoma
    group_col = 'CCAA'  # Asegúrate de que este es el nombre correcto de tu columna de comunidad autónoma
    #Crear un objeto SimpleImputer para imputación por moda
    imputer = SimpleImputer(strategy='most_frequent')
    #Función para aplicar la imputación por moda a cada grupo
    def impute_group(group):
        group[cat_cols] = imputer.fit_transform(group[cat_cols])
        return group
    #Aplicar la imputación a cada grupo de comunidad autónoma usando apply
    df = df.groupby(group_col, group_keys=False).apply(impute_group)


    #Imputación de numéricos por la mediana
    num_cols = df.select_dtypes(include=['number']).columns
    # Crear un objeto SimpleImputer con la estrategia de imputación 'median'
    imputer_num = SimpleImputer(strategy='median')
    #Función para aplicar la imputación por moda a cada grupo
    def impute_group_num(group):
        group[num_cols] = imputer_num.fit_transform(group[num_cols])
        return group
    # Aplicar la imputación solo a las columnas numéricas
    df = df.groupby(group_col, group_keys=False).apply(impute_group_num)

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

    output_path = os.path.join(script_dir, 'datamunging/consolidated_data.csv')
    df.to_csv(output_path, index=False)


def main():
    # Carpeta principal (path relativo en la ubicación local del proyecto)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_tuning(script_dir)

if __name__ == "__main__":
    main()