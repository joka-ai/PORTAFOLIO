import os
from limpieza_transformacion import limpiar_dataset

if __name__ == "__main__":
    ruta_datos = "data/netflix_titles.csv" 

    if os.path.isfile(ruta_datos):
        print(f"Archivo encontrado: {ruta_datos}.")
        
        df_clean = limpiar_dataset(ruta_datos, guardar_csv=True)
        
        print("Limpieza completada y archivo guardado.")
    else:
    
        print(f"Error: no se encontró el archivo {ruta_datos}. Intentando buscar en el directorio actual...")
        ruta_datos_alt = "netflix_titles.csv"
        
        if os.path.isfile(ruta_datos_alt):
            print(f"Archivo encontrado: {ruta_datos_alt}.")
            df_clean = limpiar_dataset(ruta_datos_alt, guardar_csv=False)
            print("Limpieza completada. No se guardó el CSV porque la ruta 'data/' falló.")
        else:
            print(f"Error: no se encontró el archivo {ruta_datos_alt} tampoco. Verifica tu ruta de archivos.")