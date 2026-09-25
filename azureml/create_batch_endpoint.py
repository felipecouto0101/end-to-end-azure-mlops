"""
IaC — Criação e atualização do Batch Endpoint no Azure ML.

Lógica idempotente:
  - Endpoint não existe → cria endpoint + cria deployment
  - Endpoint já existe  → só atualiza o deployment com a versão mais recente do modelo

Pode ser executado quantas vezes quiser sem efeitos colaterais.
"""

# ── Constantes ────────────────────────────────────────────────────────────────
ENDPOINT_NAME   = "diabetes-batch-endpoint"
DEPLOYMENT_NAME = "diabetes-batch-dp"
MODEL_NAME      = "diabetes-rf-model"
COMPUTE_NAME    = "cluster-diabetes"


def get_latest_model_version(ml_client, model_name: str) -> str:
    """Retorna a versão mais recente do modelo registado no Model Registry."""
    versions = list(ml_client.models.list(name=model_name))
    if not versions:
        raise ValueError(
            f"Nenhuma versão do modelo '{model_name}' encontrada no registry."
        )
    latest = max(versions, key=lambda m: int(m.version))
    return latest.version


def endpoint_exists(ml_client, endpoint_name: str) -> bool:
    """Verifica se o Batch Endpoint já existe no workspace."""
    try:
        ml_client.batch_endpoints.get(endpoint_name)
        return True
    except Exception:
        return False


def create_or_update_batch_endpoint(ml_client) -> str:
    """
    Cria o endpoint se não existir.
    Cria ou atualiza o deployment com a versão mais recente do modelo.

    Retorna o scoring_uri do endpoint.
    """
    from azure.ai.ml.entities import (
        BatchEndpoint,
        ModelBatchDeployment,
        ModelBatchDeploymentSettings,
        BatchRetrySettings,
        CodeConfiguration,
        Environment,
    )
    from azure.ai.ml.constants import BatchDeploymentOutputAction

    # 1. Obter versão mais recente do modelo
    model_version = get_latest_model_version(ml_client, MODEL_NAME)
    model_id = f"azureml:{MODEL_NAME}:{model_version}"
    print(f"Modelo selecionado: {MODEL_NAME} @ versão {model_version}")

    # 2. Criar endpoint se não existir
    if not endpoint_exists(ml_client, ENDPOINT_NAME):
        print(f"Endpoint '{ENDPOINT_NAME}' não encontrado. Criando...")
        endpoint = BatchEndpoint(
            name=ENDPOINT_NAME,
            description="Batch Endpoint para predição de diabetes em larga escala",
            tags={"model": MODEL_NAME, "project": "end-to-end-azure-mlops"},
        )
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint '{ENDPOINT_NAME}' criado com sucesso.")
    else:
        print(f"Endpoint '{ENDPOINT_NAME}' já existe. Atualizando deployment...")

    # 3. Ambiente para o deployment
    batch_env = Environment(
        name="diabetes-batch-env",
        conda_file="azureml/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )

    # 4. Criar ou atualizar o deployment com ModelBatchDeployment
    deployment = ModelBatchDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model_id,
        compute=COMPUTE_NAME,
        code_configuration=CodeConfiguration(
            code="./src",
            scoring_script="batch_driver.py",
        ),
        environment=batch_env,
        settings=ModelBatchDeploymentSettings(
            instance_count=1,
            max_concurrency_per_instance=2,
            mini_batch_size=10,
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            output_file_name="predictions.csv",
            retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
        ),
        tags={
            "model_version": model_version,
            "model_name": MODEL_NAME,
        },
    )

    print(f"Criando/atualizando deployment '{DEPLOYMENT_NAME}'...")
    ml_client.batch_deployments.begin_create_or_update(deployment).result()

    # 5. Definir como deployment padrão do endpoint
    try:
        endpoint = ml_client.batch_endpoints.get(ENDPOINT_NAME)
        endpoint.defaults.deployment_name = DEPLOYMENT_NAME
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    except Exception as e:
        print(f"Aviso: não foi possível definir deployment padrão: {e}")
        print("O deployment foi criado mas pode não estar definido como padrão.")

    # 6. Obter URI do endpoint
    endpoint = ml_client.batch_endpoints.get(ENDPOINT_NAME)
    scoring_uri = endpoint.scoring_uri

    print(f"\nBatch Endpoint pronto!")
    print(f"  Nome       : {ENDPOINT_NAME}")
    print(f"  Deployment : {DEPLOYMENT_NAME}")
    print(f"  Modelo     : {MODEL_NAME} @ v{model_version}")
    print(f"  URI        : {scoring_uri}")

    return scoring_uri


if __name__ == "__main__":
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    print(f"Conectado ao Workspace: {ml_client.workspace_name}")

    create_or_update_batch_endpoint(ml_client)
