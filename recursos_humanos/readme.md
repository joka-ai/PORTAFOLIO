# 📊 Proyecto de Análisis y Predicción de Rotación de Empleados (Attrition)

## 🎯 Objetivo Principal
Este proyecto implementa un pipeline completo de Data Science para analizar, visualizar y predecir la rotación de empleados (Attrition) utilizando el dataset de Recursos Humanos de IBM.

**Pregunta principal:**
> *¿Qué factores están impulsando la rotación de empleados (Attrition) y cómo podemos predecir qué empleados son los más propensos a irse?*

---

## 🔍 Desglose del Análisis (Los 5 Porqués)

1. **¿Por qué se van los empleados?**  
   Análisis demográfico y de experiencia: edad, antigüedad, experiencia total.

2. **¿Por qué ciertos departamentos/roles tienen mayor rotación?**  
   Segmentación por área, rol y funciones específicas.

3. **¿Por qué influyen las percepciones del empleado?**  
   Impacto de la satisfacción ambiental, del puesto y del balance vida-trabajo.

4. **¿Por qué la compensación no es suficiente?**  
   Relación entre salario, aumentos y horas extra.

5. **¿Por qué podemos predecir la rotación?**  
   Identificación de factores clave para modelado predictivo con Random Forest.

---

## ⚙️ Estructura y Metodología del Pipeline

El proyecto se organiza en una arquitectura modular dentro de `src/`, garantizando claridad, reproducibilidad y robustez en el modelado.

### Fases del Pipeline

| Fase | Archivo | Descripción |
|------|---------|-------------|
| **Limpieza** | `src/limpieza.py` | Carga del CSV, eliminación de columnas constantes (`EmployeeCount`, `StandardHours`), imputación de nulos. |
| **EDA y Estadísticas** | `src/eda.py` | Cálculo de la tasa de rotación, análisis por departamento, edad, ingresos y satisfacción. Gráficos e informes exportados. |
| **Modelado** | `src/modelado.py` | Construcción de `ColumnTransformer`, aplicación de SMOTE, tuning con Optuna y entrenamiento del modelo Random Forest. |
| **Orquestación** | `run_pipeline.py` | Orquesta todas las etapas leyendo `config.yaml` y guardando los resultados en `outputs/`. |

---

## 📈 Resultados y Conclusiones

### 1. Hallazgos Descriptivos (Porqués 1–4)

- **Compensación y esfuerzo:**  
  `OverTime` es un predictor clave; quienes se van trabajan más horas extra y ganan menos.

- **Segmentos con mayor riesgo:**  
  Alta rotación en el área de Ventas y entre empleados solteros.

- **Experiencia:**  
  Empleados más jóvenes y con menos años en la empresa presentan tasas más altas de rotación.

### 2. Hallazgos Predictivos (Porqué 5)

- **Modelo:** Random Forest  
- **Rendimiento:** `ROC AUC = 0.7786`  
- **Variables más importantes según SHAP:**  
  - `OverTime`  
  - `MonthlyIncome`  
  - `Age`  
  - `JobRole_Laboratory Technician`

### Conclusión General

El modelo demuestra que la rotación es **altamente predecible**.  
Las acciones de retención deberían enfocarse en:
- Reducir horas extra,
- Revisar compensación en roles vulnerables,
- Diseñar estrategias para talento joven y soltero.

---

## ▶️ Ejecución del Pipeline

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --config config.yaml

