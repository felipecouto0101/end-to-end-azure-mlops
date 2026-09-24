"""
Testes de qualidade de dados usando Great Expectations 1.x.
Valida o dataset antes do treinamento do modelo.
Executar com: pytest tests/test_data_quality.py -v
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
import great_expectations as gx

# Adiciona src/ ao path para importar train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import COLNAMES


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """
    Tenta carregar o dataset real. Se não estiver disponível,
    usa um dataset sintético com a mesma estrutura.
    """
    real_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    try:
        data = pd.read_csv(real_url, names=COLNAMES)
        print(f"\nDataset real carregado: {len(data)} linhas")
        return data
    except Exception:
        print("\nDataset real indisponível — usando dataset sintético.")
        np.random.seed(42)
        n = 768
        return pd.DataFrame({
            "Pregnancies": np.random.randint(0, 15, n),
            "Glucose": np.random.randint(60, 200, n),
            "BloodPressure": np.random.randint(40, 120, n),
            "SkinThickness": np.random.randint(0, 60, n),
            "Insulin": np.random.randint(0, 400, n),
            "BMI": np.round(np.random.uniform(18.0, 50.0, n), 1),
            "DiabetesPedigreeFunction": np.round(np.random.uniform(0.05, 2.5, n), 3),
            "Age": np.random.randint(18, 80, n),
            "Outcome": np.random.randint(0, 2, n),
        })


@pytest.fixture(scope="module")
def validator(df):
    """Cria um Validator do GE 1.x a partir do DataFrame."""
    context = gx.get_context(mode="ephemeral")

    datasource = context.data_sources.add_pandas("diabetes_source")
    asset = datasource.add_dataframe_asset("diabetes_asset")
    batch_request = asset.build_batch_request(options={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name="diabetes_suite"))

    return context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="diabetes_suite",
    )


# ── Testes: Estrutura ──────────────────────────────────────────────────────────

def test_dataset_has_all_columns(validator):
    """O dataset deve conter todas as 9 colunas esperadas."""
    for col in COLNAMES:
        result = validator.expect_column_to_exist(col)
        assert result.success, f"Coluna ausente: {col}"


def test_dataset_minimum_rows(validator):
    """O dataset deve ter pelo menos 500 linhas."""
    result = validator.expect_table_row_count_to_be_between(min_value=500)
    assert result.success, f"Dataset muito pequeno: {result.result}"


def test_dataset_column_count(validator):
    """O dataset deve ter exatamente 9 colunas."""
    result = validator.expect_table_column_count_to_equal(value=9)
    assert result.success, f"Número de colunas incorreto: {result.result}"


# ── Testes: Valores Nulos ──────────────────────────────────────────────────────

def test_no_null_values_in_critical_columns(validator):
    """As colunas críticas não devem ter valores nulos."""
    for col in ["Glucose", "BMI", "Age", "Outcome"]:
        result = validator.expect_column_values_to_not_be_null(col)
        assert result.success, f"Coluna '{col}' contém valores nulos"


def test_null_rate_below_threshold(validator):
    """Nenhuma coluna deve ter mais de 5% de valores nulos."""
    for col in COLNAMES:
        result = validator.expect_column_values_to_not_be_null(col, mostly=0.95)
        assert result.success, f"Coluna '{col}' tem mais de 5% de nulos"


# ── Testes: Coluna Alvo (Outcome) ──────────────────────────────────────────────

def test_outcome_is_binary(validator):
    """Outcome deve conter apenas os valores 0 e 1."""
    result = validator.expect_column_values_to_be_in_set("Outcome", [0, 1])
    assert result.success, "Outcome contém valores fora de {0, 1}"


def test_outcome_has_both_classes(validator):
    """O dataset deve ter exemplos de ambas as classes."""
    result = validator.expect_column_distinct_values_to_contain_set(
        "Outcome", {0, 1}
    )
    assert result.success, "Outcome não contém ambas as classes"


# ── Testes: Faixas de Valores Clínicos ────────────────────────────────────────

def test_glucose_range(validator):
    """Glucose deve estar entre 0 e 300 mg/dL."""
    result = validator.expect_column_values_to_be_between(
        "Glucose", min_value=0, max_value=300, mostly=0.99
    )
    assert result.success, f"Glucose fora do intervalo clínico: {result.result}"


def test_bmi_range(validator):
    """BMI deve estar entre 10 e 70 (permite até 2% de zeros codificando missings)."""
    result = validator.expect_column_values_to_be_between(
        "BMI", min_value=10.0, max_value=70.0, mostly=0.98
    )
    assert result.success, f"BMI fora do intervalo esperado: {result.result}"


def test_age_range(validator):
    """Age deve estar entre 1 e 120 anos."""
    result = validator.expect_column_values_to_be_between(
        "Age", min_value=1, max_value=120, mostly=0.99
    )
    assert result.success, f"Age fora do intervalo esperado: {result.result}"


def test_pregnancies_non_negative(validator):
    """Pregnancies não deve ter valores negativos."""
    result = validator.expect_column_values_to_be_between(
        "Pregnancies", min_value=0, max_value=20, mostly=0.99
    )
    assert result.success, f"Pregnancies contém valores inválidos: {result.result}"


def test_blood_pressure_range(validator):
    """BloodPressure deve estar entre 0 e 200 mmHg."""
    result = validator.expect_column_values_to_be_between(
        "BloodPressure", min_value=0, max_value=200, mostly=0.99
    )
    assert result.success, f"BloodPressure fora do intervalo: {result.result}"


# ── Testes: Tipos de Dados ─────────────────────────────────────────────────────

def test_numeric_columns_types(validator):
    """Todas as colunas devem ser numéricas."""
    for col in COLNAMES:
        result = validator.expect_column_values_to_be_in_type_list(
            col, ["int64", "float64", "int32", "float32", "int8", "int16"]
        )
        assert result.success, f"Coluna '{col}' não é numérica"
