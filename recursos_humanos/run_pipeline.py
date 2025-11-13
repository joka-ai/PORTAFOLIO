"""run_pipeline.py - entrypoint to run the full pipeline using the generated modules"""
import argparse
import logging
from pathlib import Path
import yaml

from src.utils import setup_logging, ensure_dirs, version_name, read_config
from src.limpieza import cargar_csv, limpiar_dataset
from src.eda import resumen_basico, graficos_basicos
from src.modelado import train_and_tune

def main(config_path: str):
    # logging & config
    setup_logging()
    logging.info(f"Leyendo config desde {config_path}")
    config = read_config(config_path)

    data_path = config["data"]["input_path"]
    outputs_dir = config["outputs"]["dir"]
    models_dir = config["outputs"]["models_dir"]
    logs_dir = config["outputs"]["logs_dir"]
    ensure_dirs([outputs_dir, models_dir, logs_dir])

    # Load & clean
    df = cargar_csv(data_path)
    df_clean = limpiar_dataset(df)

    # EDA
    resumen_basico(df_clean, outputs_dir)
    graficos_basicos(df_clean, outputs_dir)

    # Modeling
    X = df_clean.drop(columns=["Attrition"])
    y = df_clean["Attrition"]
    result = train_and_tune(X, y, models_dir, config["modeling"])

    # manifest / simple versioning metadata
    manifest = {
        "model_version": version_name(config["modeling"].get("model_prefix", "attrition_model")),
        "model_artifact": result["model_path"],
        "auc": result["auc"],
        "shap_plot": result.get("shap_path"),
    }
    Path(outputs_dir).joinpath("manifest.yaml").write_text(yaml.safe_dump(manifest))
    logging.info("Pipeline completed")
    logging.info(f"Manifest written to {Path(outputs_dir)/'manifest.yaml'}")
    print("Pipeline finalizado. Revisar outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Ruta al config.yaml")
    args = parser.parse_args()
    main(args.config)
