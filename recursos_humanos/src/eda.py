"""eda.py - exploratory data analysis and visualizations"""
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def resumen_basico(df: pd.DataFrame, outputs_dir: str):
    logging.info("Generando estadísticas descriptivas")
    desc = df.describe(include="all").transpose()
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    desc.to_csv(Path(outputs_dir) / "estadisticas_descriptivas.csv")

    tasa = df["Attrition"].mean()
    with open(Path(outputs_dir) / "resumen_attrition.txt", "w") as f:
        f.write(f"Tasa global de Attrition: {tasa:.4f}\n")

def graficos_basicos(df: pd.DataFrame, outputs_dir: str):
    logging.info("Generando visualizaciones")
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    # Distribucion Attrition
    fig, ax = plt.subplots()
    sns.countplot(x="Attrition", data=df, ax=ax)
    ax.set_title("Distribución de Attrition (0=No,1=Sí)")
    save_fig(fig, Path(outputs_dir) / "attrition_counts.png")

    # Attrition por Department
    if "Department" in df.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        dept = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False)
        sns.barplot(x=dept.index, y=dept.values, ax=ax)
        ax.set_ylabel("Tasa Attrition")
        ax.set_title("Tasa de Attrition por Departamento")
        ax.tick_params(axis="x", rotation=45)
        save_fig(fig, Path(outputs_dir) / "attrition_by_department.png")

    # Income vs Attrition boxplot
    if "MonthlyIncome" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, ax=ax)
        ax.set_title("MonthlyIncome por Attrition")
        save_fig(fig, Path(outputs_dir) / "monthlyincome_by_attrition.png")
