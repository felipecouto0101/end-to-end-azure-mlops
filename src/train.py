import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Caminho do recurso de dados")
    args = parser.parse_args()

    # Iniciar rastreamento automático do MLflow no Azure ML
    mlflow.autolog()

    print(f"Carregando dados de: {args.data_path}")
    colnames = [
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
    df = pd.read_csv(args.data_path, names=colnames)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Treinando o modelo RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Acurácia do modelo: {acc:.4f}")
    print(f"AUC ROC: {auc:.4f}")

    # Log do modelo no registro de artefatos
    mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
