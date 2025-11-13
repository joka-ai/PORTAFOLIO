"""modelado.py - modeling, tuning, interpretability, persistence"""
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Opcionales (si están instalados)
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

try:
    import optuna
except Exception:
    optuna = None

try:
    import shap
except Exception:
    shap = None

try:
    import mlflow
    mlflow_available = True
except Exception:
    mlflow_available = False

def prepare_features(X: pd.DataFrame):
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ], remainder="drop")

    return preprocessor, num_features, cat_features

def train_and_tune(X: pd.DataFrame, y: pd.Series, outputs_dir: str, config: dict):
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    rs = config.get("random_state", 42)
    test_size = config.get("test_size", 0.25)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )

    preprocessor, num_feats, cat_feats = prepare_features(X)

    # SMOTE
    if config.get("smote", True) and SMOTE is not None:
        logging.info("Aplicando SMOTE en conjunto de entrenamiento")
        sm = SMOTE(random_state=rs)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        logging.info("SMOTE no disponible o desactivado; usando datos originales")
        X_train_res, y_train_res = X_train, y_train

    # Pipeline base
    pipeline = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=rs))])

    # Optuna tuning (opcional)
    if optuna is not None and config.get("optuna", True):
        logging.info("Optuna disponible: iniciando tuning")
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=rs)
            pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
            pipe.fit(X_train_res, y_train_res)
            proba = pipe.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, proba)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.get("n_trials", 20))
        best_params = study.best_params
        logging.info(f"Optuna best params: {best_params}")
        pipeline = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(**best_params, random_state=rs))])

    # Fit final
    pipeline.fit(X_train_res, y_train_res)

    # Evaluate
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds)
    logging.info(f"ROC AUC en test: {auc:.4f}")

    # Persistir modelo
    model_name = f"model_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.joblib"
    model_path = Path(outputs_dir) / model_name
    joblib.dump(pipeline, model_path)
    logging.info(f"Saved model to {model_path}")

    # MLflow optional
    if mlflow_available:
        try:
            mlflow.sklearn.log_model(pipeline, "model")
            logging.info("Logged model with MLflow")
        except Exception as e:
            logging.warning(f"Could not log to mlflow: {e}")

    # SHAP optional (muestra pequeña)
    shap_path = None
    if shap is not None:
        try:
            logging.info("Generando SHAP explainer (muestra pequeña)")
            # usamos transform+clf directamente para evitar problemas con pipeline + shap
            X_sample = X_train_res.sample(min(200, len(X_train_res)), random_state=rs)
            transformed = pipeline.named_steps["pre"].transform(X_sample)
            clf = pipeline.named_steps["clf"]
            explainer = shap.Explainer(clf, transformed, feature_perturbation="tree_path_dependent")
            X_test_sample = X_test.sample(min(200, len(X_test)), random_state=rs)
            transformed_test = pipeline.named_steps["pre"].transform(X_test_sample)
            shap_vals = explainer(transformed_test)
            shap_path = Path(outputs_dir) / "shap_summary.png"
            shap.plots.beeswarm(shap_vals, show=False)
            import matplotlib.pyplot as plt
            plt.savefig(shap_path, bbox_inches="tight")
            plt.close()
            logging.info(f"SHAP plot saved to {shap_path}")
        except Exception as e:
            logging.warning(f"SHAP failed: {e}")

    return {
        "model_path": str(model_path),
        "auc": float(auc),
        "report": report,
        "shap_path": str(shap_path) if shap_path is not None else None,
    }
