from limpieza_transformacion import limpiar_dataset

if __name__ == "__main__":
    ruta_datos = "data/netflix_titles.csv"
    df_clean = limpiar_dataset(ruta_datos, guardar_csv=True)
  

