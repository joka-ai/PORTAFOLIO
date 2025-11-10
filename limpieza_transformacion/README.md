# 📊 Limpieza de Datos – Netflix Dataset

Este proyecto se centra en la **limpieza y transformación de datos** del catálogo de Netflix, preparando el dataset para futuros análisis y visualizaciones.

Dataset: [`netflix_titles.csv` – Netflix Movies and TV Shows (Kaggle)](https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download)

---

## 🔹 Objetivo
Limpiar y normalizar los datos del catálogo de Netflix para asegurar consistencia, legibilidad y trazabilidad de los valores imputados.

---

## 🔹 Qué se hizo

1. **Detección y corrección de valores inconsistentes**
   - Algunos valores de duración estaban en la columna `rating`.
   - Se trasladaron correctamente a `duration` cuando esta estaba vacía.

2. **Limpieza de texto**
   - Columnas como `country`, `title`, `cast`, `director`, `listed_in` y `description` fueron normalizadas:
     - Convertidas a minúsculas.
     - Espacios innecesarios eliminados.
     - Valores faltantes marcados como `unknown`.

3. **Conversión de fechas**
   - La columna `date_added` se transformó a tipo datetime para análisis temporal.

4. **Valores faltantes**
   - Valores que no se podían interpretar se pusieron como `unknown`.
   - Columnas críticas como `rating` y `date_added` fueron **imputadas usando la moda** por `release_year` y `type` (película o serie):
     - Se reemplaza el valor faltante por el más frecuente dentro de cada grupo.
     - Se añade una columna booleana (`*_imputed`) para identificar qué valores fueron imputados, facilitando la diferenciación en análisis posteriores.

5. **Normalización de listas**
   - Columnas como `country` y `listed_in` estaban mal formateadas:
     - Se separan los valores por comas, se eliminan duplicados y se ordenan alfabéticamente.
---

## 🔹 Resultado
- Dataset limpio, con columnas consistentes y valores faltantes correctamente imputados.
- Trazabilidad de valores imputados para análisis transparente y explicable.

---

## 🔹 Tecnologías utilizadas
- Python 3
- Pandas
- NumPy

