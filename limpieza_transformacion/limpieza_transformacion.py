"""
    LO QUE SE VIO EN LA VISUALIZACION Y LIMPIEZA:

1 - HAY VALORES DE DURACION EN LA COLUMNA 'rating':  
    - Si los valores de 'duration' estan vacios, se corrigen tomando lo que estaba en 'rating'.

2 - LIMPIEZA DE TEXTOS:  
    - Se normalizan columnas de texto ('country', 'title', 'cast', 'director', 'listed_in', 'description')  
      para que sean minusculas, sin espacios innecesarios y legibles.

3 - CONVERSION DE FECHAS:  
    - La columna 'date_added' se convierte a tipo fecha (datetime) para poder analizar cronologicamente.

4 - VALORES DESCONOCIDOS:  
    - Todo valor que no se pueda interpretar se pone como 'unknown'.

5 - IMPUTACION CON MODA:  
    - Columnas como 'rating' y 'date_added' pueden tener valores faltantes.  
    - Para no perder informacion, se imputan usando la **moda** agrupada por 'release_year' y 'type' (pelicula o serie).  
    - Esto significa que se reemplaza el valor faltante por el más frecuente en ese grupo, asegurando consistencia.  
    - Se marca una columna booleana ('*_imputed') para saber que el valor fue imputado,  
      permitiendo diferenciar en analisis o graficos entre datos originales y datos rellenados.

6 - NORMALIZACION DE LISTAS:  
    - Columnas como 'country' y 'listed_in' estaban mal formateadas.  
    - Se separan por comas, se eliminan duplicados y se ordenan alfabeticamente para un analisis correcto.
"""

import pandas as pd
import numpy as np
import re

def visualizacion(df):
    print('primeras 5 filas',df.head())
    print('ultimas 5 filas',df.tail())
    print('tipos de datos',df.dtypes)
    print('cantidad de datos:',df.shape)
    print("valores nulos por columna:",df.isnull().sum())
    print("cadenas vacias por columna:",(df == '').sum())
    print("cantidad de filas duplicadas:",df.duplicated().sum())
    print("cantidad de duplicados 'show_id':",df['show_id'].duplicated().sum())
    print("type:", df['type'].dropna().unique())
    print("rating:", df['rating'].dropna().unique())
    print("listed_in:", df['listed_in'].dropna().unique())
    print("country:", df['country'].dropna().unique())


def limpiar_texto(df):
    cols_texto = ['country', 'title', 'cast', 'director', 'listed_in', 'description']
    for col in cols_texto:
        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    df['rating'] = df['rating'].fillna('unknown')
    return df

def normalizar_listas(df):
    cols_multi = ['country', 'listed_in']

    for col in cols_multi:
        df[col] = df[col].apply(
            lambda x: (
                ', '.join(
                    sorted(set(i.strip() for i in x.split(',') if i.strip()))
                )
                if x != 'unknown' else 'unknown'
            )
        )

    return df

def corregir_rating_minutos(df):
    mask_minutos = df['rating'].isin(['74 min', '84 min', '66 min'])
    print("filas con raiting en minutos:")
    print(df.loc[mask_minutos, ['show_id', 'duration', 'rating']])

    df.loc[mask_minutos & (df['duration'].isin(['', 'unknown', np.nan])), 'duration'] = df.loc[mask_minutos, 'rating']
    df.loc[mask_minutos, 'rating'] = 'unknown'
    return df


def imputar_fecha(df):
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    moda_por_anio_tipo = (
        df.dropna(subset=['date_added'])
          .groupby(['release_year', 'type'])['date_added']
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NaT)
    )

    df['date_imputed'] = False

    def _imputar(row):
        if pd.isna(row['date_added']):
            key = (row['release_year'], row['type'])
            if key in moda_por_anio_tipo.index:
                row['date_added'] = moda_por_anio_tipo.loc[key]
            elif not pd.isna(row['release_year']):
                row['date_added'] = pd.Timestamp(f"{int(row['release_year'])}-01-01")
            else:
                row['date_added'] = pd.Timestamp('1900-01-01')
            row['date_imputed'] = True
        return row

    return df.apply(_imputar, axis=1)


def normalizar_duracion(df):
    df['duration'] = df['duration'].fillna('unknown').astype(str).str.lower().str.strip()
    df['duration_value'] = np.nan
    df['duration_unit'] = 'unknown'

    pattern = r'(\d+)\s*(min|season|seasons)?'
    extracted = df['duration'].str.extract(pattern)
    df['duration_value'] = pd.to_numeric(extracted[0], errors='coerce')
    df['duration_unit'] = extracted[1].fillna('unknown')

    df.loc[df['type'].str.lower() == 'movie', 'duration_unit'] = df.loc[df['type'].str.lower() == 'movie', 'duration_unit'].replace('unknown', 'min')
    df.loc[df['type'].str.lower() == 'tv show', 'duration_unit'] = df.loc[df['type'].str.lower() == 'tv show', 'duration_unit'].replace('unknown', 'seasons')

    return df


def imputar_rating(df):
    df['rating'] = df['rating'].fillna('unknown').astype(str).str.strip().str.upper()

    moda_rating = (
        df[df['rating'] != 'UNKNOWN']
          .groupby(['release_year', 'type'])['rating']
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNKNOWN')
    )

    df['rating_imputed'] = False

    def _imputar(row):
        if row['rating'] in ['UNKNOWN', '', np.nan]:
            key = (row['release_year'], row['type'])
            if key in moda_rating.index:
                row['rating'] = moda_rating.loc[key]
            else:
                row['rating'] = 'UNKNOWN'
            row['rating_imputed'] = True
        return row

    return df.apply(_imputar, axis=1)


def limpiar_dataset(ruta_csv, guardar_csv=True):
    df = pd.read_csv(ruta_csv)
    print("VISUALIZACIÓN INICIAL")
    visualizacion(df)
    df = limpiar_texto(df)
    df = normalizar_listas(df) 
    df = corregir_rating_minutos(df)
    df = imputar_fecha(df)
    df = normalizar_duracion(df)
    df = imputar_rating(df)
    if guardar_csv:
        salida = "data/netflix_clean.csv"
        df.to_csv(salida, index=False)
    return df
