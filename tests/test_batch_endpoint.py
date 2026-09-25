"""
Testes unitários para azureml/create_batch_endpoint.py.
Usa mocks para simular o Azure ML SDK sem conexão real.
Executar com: pytest tests/test_batch_endpoint.py -v
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "azureml"))
from create_batch_endpoint import (
    get_latest_model_version,
    endpoint_exists,
    ENDPOINT_NAME,
    DEPLOYMENT_NAME,
    MODEL_NAME,
    COMPUTE_NAME,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_mock_model(version: str):
    """Cria um mock de modelo com a versão especificada."""
    m = MagicMock()
    m.version = version
    return m


def make_mock_client(model_versions=None, endpoint_exists_flag=True):
    """Cria um mock do MLClient configurado para os testes."""
    client = MagicMock()

    # mock models.list — usa lista vazia se explicitamente passada
    if model_versions is None:
        model_versions = ["1", "2", "3"]
    client.models.list.return_value = [make_mock_model(v) for v in model_versions]

    # mock batch_endpoints.get
    if endpoint_exists_flag:
        client.batch_endpoints.get.return_value = MagicMock(
            scoring_uri="https://fake-endpoint.azure.com/score"
        )
    else:
        client.batch_endpoints.get.side_effect = Exception("Endpoint not found")

    # mock begin_create_or_update retorna um poller com .result()
    poller = MagicMock()
    poller.result.return_value = None
    client.batch_endpoints.begin_create_or_update.return_value = poller
    client.batch_deployments.begin_create_or_update.return_value = poller

    return client


# ── Testes: get_latest_model_version ──────────────────────────────────────────

def test_get_latest_model_version_returns_highest():
    """Deve retornar a maior versão disponível."""
    client = make_mock_client(model_versions=["1", "2", "3"])
    version = get_latest_model_version(client, MODEL_NAME)
    assert version == "3"


def test_get_latest_model_version_single_version():
    """Com apenas uma versão, deve retorná-la."""
    client = make_mock_client(model_versions=["1"])
    version = get_latest_model_version(client, MODEL_NAME)
    assert version == "1"


def test_get_latest_model_version_unordered():
    """Deve retornar a maior versão mesmo fora de ordem."""
    client = make_mock_client(model_versions=["3", "1", "5", "2"])
    version = get_latest_model_version(client, MODEL_NAME)
    assert version == "5"


def test_get_latest_model_version_raises_when_no_models():
    """Deve lançar ValueError quando não há modelos registados."""
    client = make_mock_client(model_versions=[])
    with pytest.raises(ValueError, match="Nenhuma"):
        get_latest_model_version(client, MODEL_NAME)


def test_get_latest_model_version_calls_correct_model_name():
    """Deve chamar models.list com o nome correto do modelo."""
    client = make_mock_client(model_versions=["1"])
    get_latest_model_version(client, MODEL_NAME)
    client.models.list.assert_called_once_with(name=MODEL_NAME)


# ── Testes: endpoint_exists ────────────────────────────────────────────────────

def test_endpoint_exists_returns_true_when_found():
    """Deve retornar True quando o endpoint existe."""
    client = make_mock_client(endpoint_exists_flag=True)
    assert endpoint_exists(client, ENDPOINT_NAME) is True


def test_endpoint_exists_returns_false_when_not_found():
    """Deve retornar False quando o endpoint não existe."""
    client = make_mock_client(endpoint_exists_flag=False)
    assert endpoint_exists(client, ENDPOINT_NAME) is False


def test_endpoint_exists_calls_correct_endpoint_name():
    """Deve chamar batch_endpoints.get com o nome correto."""
    client = make_mock_client(endpoint_exists_flag=True)
    endpoint_exists(client, ENDPOINT_NAME)
    client.batch_endpoints.get.assert_called_once_with(ENDPOINT_NAME)


# ── Testes: constantes ─────────────────────────────────────────────────────────

def test_endpoint_name_is_defined():
    """ENDPOINT_NAME deve estar definido."""
    assert isinstance(ENDPOINT_NAME, str)
    assert len(ENDPOINT_NAME) > 0


def test_deployment_name_is_defined():
    """DEPLOYMENT_NAME deve estar definido."""
    assert isinstance(DEPLOYMENT_NAME, str)
    assert len(DEPLOYMENT_NAME) > 0


def test_model_name_matches_registry():
    """MODEL_NAME deve corresponder ao modelo registado no pipeline."""
    assert MODEL_NAME == "diabetes-rf-model"


def test_compute_name_matches_cluster():
    """COMPUTE_NAME deve corresponder ao cluster existente."""
    assert COMPUTE_NAME == "cluster-diabetes"
