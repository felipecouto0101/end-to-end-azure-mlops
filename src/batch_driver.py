"""
Scoring script para o Batch Endpoint.
Carrega o modelo e gera predições para cada mini-batch de dados.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd


def init():
    """Carrega o modelo quando o deployment arranca."""
    global model
    model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR", "."), "model.pkl")
    model = joblib.load(model_path)
    print(f"Modelo carregado de: {model_path}")


def run(mini_batch):
    """
    Processa um mini-batch de ficheiros CSV e retorna predições.

    Args:
        mini_batch: lista de caminhos para ficheiros CSV

    Returns:
        DataFrame com predições
    """
    COLNAMES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    results = []

    for file_path in mini_batch:
        try:
            df = pd.read_csv(file_path, names=COLNAMES)
            X = df.drop("Outcome", axis=1, errors="ignore")
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "file": os.path.basename(file_path),
                    "row": i,
                    "prediction": int(pred),
                    "probability": round(float(prob), 4),
                })
        except Exception as e:
            print(f"Erro ao processar {file_path}: {e}")

    return pd.DataFrame(results)
