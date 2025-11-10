import pandas as pd
import numpy as np
import re

# Función para explorar 
def visualizacion(df):
    print('Primeras 5 filas:')
    print(df.head())
    print('Últimas 5 filas:')
    print(df.tail())
    print('Tipos de datos por columna:')
    print(df.dtypes)
    print('Cantidad de filas y columnas:', df.shape)
    print('Valores nulos por columna:')
    print(df.isnull().sum())
    print('Cantidad de filas duplicadas:', df.duplicated().sum())
    print("Cantidad de duplicados en 'show_id':", df['show_id'].duplicated().sum())
    print('Tipos de contenido únicos:', df['type'].dropna().unique())
    print('Ratings únicos:', df['rating'].dropna().unique())
    
# Limpieza y normalización de textos
def limpiar_texto(df):
    cols_texto = ['country', 'title', 'cast', 'director', 'listed_in', 'description']
    for col in cols_texto:
        # Rellenar valores nulos con 'unknown', convertir a string, eliminar espacios y pasar a minúsculas
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    # Asegurarse que 'rating' también tenga valores válidos
    df['rating'] = df['rating'].fillna('unknown')
    return df


# Normaliza columnas que contienen listas separadas por comas (Corregida)
def normalizar_listas(df):
    cols_multi = ['country', 'listed_in', 'director', 'cast'] 
    for col in cols_multi:
        df[col] = df[col].apply(
            lambda x: (
                # Se asegura la conversión a str para robustez.
                # Separa por coma, limpia espacios, elimina duplicados (set), ordena y vuelve a unir.
                ', '.join(sorted(set(i.strip() for i in str(x).split(',') if i.strip())))
                if str(x).lower().strip() != 'unknown' and str(x).strip() != '' else 'unknown'
            )
        )
    return df


# Corrige valores de 'rating' que contienen duración en minutos (Corregida)
def corregir_rating_minutos(df):
    df['rating'] = df['rating'].astype(str).str.lower()
    
    # Usar regex para identificar cualquier patrón de duración (ej. '74 min', '120 min')
    mask_minutos = df['rating'].str.contains(r'^\d+\s*min$', na=False)

    print("Filas con rating en minutos (corregidas):")
    print(df.loc[mask_minutos, ['show_id', 'duration', 'rating']])
    
    # Si la columna 'duration' está vacía o es 'unknown', usar el valor de 'rating'
    duration_empty_mask = df['duration'].isin(['', 'unknown', np.nan])
    df.loc[mask_minutos & duration_empty_mask, 'duration'] = df.loc[mask_minutos, 'rating']
    
    # Marcar 'rating' como desconocido
    df.loc[mask_minutos, 'rating'] = 'unknown'
    return df


# Imputación de fechas faltantes usando la moda por año y tipo 
def imputar_fecha(df):
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    # Calcular la moda de 'date_added' agrupando por 'release_year' y 'type'
    moda_por_anio_tipo = (
        df.dropna(subset=['date_added'])
          .groupby(['release_year', 'type'])['date_added']
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NaT)
    )
    
    df['date_imputed'] = False  # Columna booleana para marcar imputaciones

    def _imputar(row):
        # Si la fecha está vacía, imputar
        if pd.isna(row['date_added']):
            row['date_imputed'] = True
            key = (row['release_year'], row['type'])
            
            if key in moda_por_anio_tipo.index and pd.notna(moda_por_anio_tipo.loc[key]):
                row['date_added'] = moda_por_anio_tipo.loc[key]
            
            elif pd.notna(row['release_year']):
                # Si no hay moda, usar el 1 de enero del año de lanzamiento
                try:
                    # Protección: asegurarse de que el año sea convertible a int
                    row['date_added'] = pd.Timestamp(f"{int(row['release_year'])}-01-01")
                except ValueError:
                    row['date_added'] = pd.Timestamp('1900-01-01') # Fecha por defecto
            else:
                # Si ni año ni moda disponibles, usar fecha por defecto
                row['date_added'] = pd.Timestamp('1900-01-01')
                
        return row

    return df.apply(_imputar, axis=1)


# Normalización de la columna 'duration' en valor y unidad
def normalizar_duracion(df):
    # Asegurar que la columna 'duration' esté limpia antes de normalizar
    df['duration'] = df['duration'].fillna('unknown').astype(str).str.lower().str.strip()
    df['duration_value'] = np.nan
    df['duration_unit'] = 'unknown'

    # Extraer número y unidad (min, season, seasons)
    pattern = r'(\d+)\s*(min|season|seasons)?'
    extracted = df['duration'].str.extract(pattern)
    df['duration_value'] = pd.to_numeric(extracted[0], errors='coerce')
    df['duration_unit'] = extracted[1].fillna('unknown').str.replace('seasons', 'season') # Unificar 'seasons' a 'season'

    # Ajuste de unidades según tipo de contenido
    # Asignar 'min' a películas con duración desconocida y 'season' a series con duración desconocida
    df.loc[(df['type'].str.lower() == 'movie') & (df['duration_unit'] == 'unknown'), 'duration_unit'] = 'min'
    df.loc[(df['type'].str.lower() == 'tv show') & (df['duration_unit'] == 'unknown'), 'duration_unit'] = 'season'

    return df


# Imputación de valores faltantes en 'rating'
def imputar_rating(df):
    # Asegurar que se convierta a mayúsculas DESPUÉS de limpiar_texto para unificar 'unknown'
    df['rating'] = df['rating'].fillna('unknown').astype(str).str.strip().str.upper()
    
    # Calcular moda de 'rating' por año y tipo
    moda_rating = (
        df[df['rating'] != 'UNKNOWN']
          .groupby(['release_year', 'type'])['rating']
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNKNOWN')
    )
    
    df['rating_imputed'] = False

    def _imputar(row):
        # Si el rating está vacío o desconocido, imputar
        if row['rating'] in ['UNKNOWN', '', np.nan]:
            row['rating_imputed'] = True
            key = (row['release_year'], row['type'])
            
            if key in moda_rating.index and moda_rating.loc[key] != 'UNKNOWN':
                row['rating'] = moda_rating.loc[key]
            else:
                row['rating'] = 'UNKNOWN' # Mantener UNKNOWN si no se puede imputar
                
        return row

    # Advertencia: df.apply es lento. Para grandes datasets, considerar métodos vectorizados.
    return df.apply(_imputar, axis=1)


# Función principal para limpiar dataset
def limpiar_dataset(ruta_csv, guardar_csv=True):
    df = pd.read_csv(ruta_csv)
    
    # 2. LIMPIEZA DE TEXTOS
    df = limpiar_texto(df)
    
    # 6. NORMALIZACION DE LISTAS
    df = normalizar_listas(df)
    
    # 1. CORRECCION DE RATING (DURACION)
    df = corregir_rating_minutos(df)
    
    # 3. CONVERSION Y 5. IMPUTACION DE FECHAS
    df = imputar_fecha(df)
    
    # NORMALIZACION DE DURACION (VALOR Y UNIDAD)
    df = normalizar_duracion(df)
    
    # 5. IMPUTACION DE RATING
    df = imputar_rating(df)
    
    if guardar_csv:
        salida = "data/netflix_clean.csv"
    
        try:
            df.to_csv(salida, index=False)
            print(f"Dataset limpio guardado en: {salida}")
        except FileNotFoundError:
            print("Advertencia: No se pudo guardar en 'data/netflix_clean.csv'. Guardando en la carpeta actual.")
            df.to_csv("netflix_clean.csv", index=False)
            
    print("VISUALIZACIÓN FINAL (Primeras filas con limpieza aplicada)")
    print(df.head())
    print("Valores nulos después de la limpieza:")
    print(df.isnull().sum())
    
    return df
