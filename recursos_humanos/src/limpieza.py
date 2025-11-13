"""limpieza.py - data loading and cleaning routines"""
import pandas as pd
import logging
from typing import Tuple

def cargar_csv(path: str) -> pd.DataFrame:
    logging.info(f"Cargando CSV desde {path}")
    df = pd.read_csv(path)
    return df

def identificar_columnas_constantes(df: pd.DataFrame):
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normaliza nombres
    - Elimina columnas constantes típicas
    - Forza tipos numéricos
    - Imputa NA (mediana para numéricas, 'Unknown' para categóricas)
    - Codifica Attrition -> 0/1
    """
    logging.info("Iniciando limpieza del dataset")
    df = df.copy()

    # Normalizar nombres
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Quitar columnas constantes comunes si aparecen
    constantes_a_checkear = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    for c in constantes_a_checkear:
        if c in df.columns and df[c].nunique(dropna=False) <= 1:
            logging.info(f"Eliminando columna constante: {c}")
            df.drop(columns=[c], inplace=True)

    # Forzar tipos en columnas conocidas
    posibles_numericas = [
        "Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome",
        "MonthlyRate","NumCompaniesWorked","PercentSalaryHike","TotalWorkingYears",
        "TrainingTimesLastYear","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]
    for col in posibles_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputación simple
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in num_cols:
        if df[c].isna().any():
            med = df[c].median()
            logging.info(f"Imputando mediana para {c}: {med}")
            df[c].fillna(med, inplace=True)

    for c in cat_cols:
        if df[c].isna().any():
            logging.info(f"Imputando 'Unknown' para {c}")
            df[c].fillna("Unknown", inplace=True)

    # Codificar objetivo
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    else:
        raise KeyError("Columna 'Attrition' no encontrada en el dataset.")

    return df
