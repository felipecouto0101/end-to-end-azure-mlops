import argparse
import json
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

COLNAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def load_data(data_path: str) -> pd.DataFrame:
    """Carrega o dataset CSV e aplica os nomes de colunas."""
    df = pd.read_csv(data_path, names=COLNAMES)
    return df


def split_data(df: pd.DataFrame):
    """Divide o dataframe em conjuntos de treino e teste."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Treina e retorna um RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Avalia o modelo e retorna as métricas."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return {"accuracy": acc, "roc_auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Caminho do recurso de dados")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Diretório para métricas")
    args = parser.parse_args()

    # Iniciar rastreamento automático do MLflow no Azure ML
    mlflow.autolog()

    print(f"Carregando dados de: {args.data_path}")
    df = load_data(args.data_path)

    X_train, X_test, y_train, y_test = split_data(df)

    print("Treinando o modelo RandomForestClassifier...")
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    print(f"Acurácia do modelo: {metrics['accuracy']:.4f}")
    print(f"AUC ROC: {metrics['roc_auc']:.4f}")

    # Métricas explícitas via MLflow
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("roc_auc", metrics["roc_auc"])

    # Gravar métricas em ficheiro JSON para leitura fiável pelo run_pipeline.py
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Métricas guardadas em: {metrics_path}")


if __name__ == "__main__":
    main()
