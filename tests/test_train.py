"""
Testes unitários para src/train.py usando pytest.
Executar com: pytest tests/test_train.py -v
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Adiciona src/ ao path para importar train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import load_data, split_data, train_model, evaluate_model, COLNAMES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame sintético com a mesma estrutura do dataset real."""
    np.random.seed(42)
    n = 200
    data = {
        "Pregnancies": np.random.randint(0, 15, n),
        "Glucose": np.random.randint(60, 200, n),
        "BloodPressure": np.random.randint(40, 120, n),
        "SkinThickness": np.random.randint(0, 60, n),
        "Insulin": np.random.randint(0, 400, n),
        "BMI": np.round(np.random.uniform(18.0, 50.0, n), 1),
        "DiabetesPedigreeFunction": np.round(np.random.uniform(0.05, 2.5, n), 3),
        "Age": np.random.randint(18, 80, n),
        "Outcome": np.random.randint(0, 2, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def trained_model(sample_df):
    """Modelo já treinado para ser reutilizado nos testes de avaliação."""
    X_train, _, y_train, _ = split_data(sample_df)
    return train_model(X_train, y_train)


# ── Testes: load_data ──────────────────────────────────────────────────────────

def test_load_data_returns_dataframe(tmp_path, sample_df):
    """load_data deve retornar um DataFrame."""
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False, header=False)
    df = load_data(str(csv_path))
    assert isinstance(df, pd.DataFrame)


def test_load_data_columns(tmp_path, sample_df):
    """load_data deve aplicar os nomes de colunas corretos."""
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False, header=False)
    df = load_data(str(csv_path))
    assert list(df.columns) == COLNAMES


def test_load_data_not_empty(tmp_path, sample_df):
    """load_data não deve retornar DataFrame vazio."""
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False, header=False)
    df = load_data(str(csv_path))
    assert len(df) > 0


# ── Testes: split_data ─────────────────────────────────────────────────────────

def test_split_data_returns_four_parts(sample_df):
    """split_data deve retornar exatamente 4 conjuntos."""
    result = split_data(sample_df)
    assert len(result) == 4


def test_split_data_proportions(sample_df):
    """O conjunto de teste deve ter ~20% dos dados."""
    X_train, X_test, _, _ = split_data(sample_df)
    total = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total
    assert 0.18 <= test_ratio <= 0.22


def test_split_data_no_outcome_in_features(sample_df):
    """A coluna Outcome não deve estar nas features."""
    X_train, X_test, _, _ = split_data(sample_df)
    assert "Outcome" not in X_train.columns
    assert "Outcome" not in X_test.columns


def test_split_data_feature_count(sample_df):
    """Deve haver exatamente 8 features (todas as colunas menos Outcome)."""
    X_train, _, _, _ = split_data(sample_df)
    assert X_train.shape[1] == 8


# ── Testes: train_model ────────────────────────────────────────────────────────

def test_train_model_returns_classifier(sample_df):
    """train_model deve retornar um RandomForestClassifier."""
    X_train, _, y_train, _ = split_data(sample_df)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_train_model_hyperparameters(sample_df):
    """O modelo deve ter os hiperparâmetros definidos no train.py."""
    X_train, _, y_train, _ = split_data(sample_df)
    model = train_model(X_train, y_train)
    assert model.n_estimators == 100
    assert model.max_depth == 5
    assert model.random_state == 42


def test_train_model_is_fitted(sample_df):
    """O modelo deve estar treinado (ter o atributo estimators_)."""
    X_train, _, y_train, _ = split_data(sample_df)
    model = train_model(X_train, y_train)
    assert hasattr(model, "estimators_")


# ── Testes: evaluate_model ─────────────────────────────────────────────────────

def test_evaluate_model_returns_dict(trained_model, sample_df):
    """evaluate_model deve retornar um dicionário."""
    _, X_test, _, y_test = split_data(sample_df)
    metrics = evaluate_model(trained_model, X_test, y_test)
    assert isinstance(metrics, dict)


def test_evaluate_model_has_required_keys(trained_model, sample_df):
    """O dicionário de métricas deve conter accuracy e roc_auc."""
    _, X_test, _, y_test = split_data(sample_df)
    metrics = evaluate_model(trained_model, X_test, y_test)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics


def test_evaluate_model_accuracy_range(trained_model, sample_df):
    """A acurácia deve estar entre 0 e 1."""
    _, X_test, _, y_test = split_data(sample_df)
    metrics = evaluate_model(trained_model, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_model_auc_range(trained_model, sample_df):
    """O AUC ROC deve estar entre 0 e 1."""
    _, X_test, _, y_test = split_data(sample_df)
    metrics = evaluate_model(trained_model, X_test, y_test)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
