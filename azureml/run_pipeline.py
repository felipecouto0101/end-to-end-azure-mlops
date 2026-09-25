import sys

# ── Limiares de qualidade para promoção do modelo ─────────────────────────────
MIN_AUC_ROC  = 0.82   # métrica principal — dataset desbalanceado
MIN_ACCURACY = 0.74   # métrica secundária


def evaluate_quality_gate(
    accuracy: float,
    roc_auc: float,
    min_accuracy: float = MIN_ACCURACY,
    min_auc_roc: float = MIN_AUC_ROC,
) -> tuple:
    """
    Verifica se as métricas do modelo atingem os limiares mínimos.

    Retorna:
        (passed, reasons) — passed=True se aprovado, reasons lista os motivos de reprovação.
    """
    reasons = []

    if roc_auc < min_auc_roc:
        reasons.append(f"AUC ROC {roc_auc:.4f} abaixo do limiar {min_auc_roc}")

    if accuracy < min_accuracy:
        reasons.append(f"Acurácia {accuracy:.4f} abaixo do limiar {min_accuracy}")

    return len(reasons) == 0, reasons


if __name__ == "__main__":
    from azure.ai.ml import MLClient, command, Input
    from azure.ai.ml.entities import Environment, Model
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential
    from mlflow.tracking import MlflowClient

    # 1. Autenticação no Workspace do Azure ML
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    print(f"Conectado com sucesso ao Workspace: {ml_client.workspace_name}")

    # 2. Definição do Ambiente
    custom_env = Environment(
        name="diabetes-ml-env",
        description="Ambiente para treinamento do modelo de diabetes",
        tags={"scikit-learn": "1.3.0"},
        conda_file="azureml/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )

    # 3. Definição do Job de Treinamento
    job = command(
        code="./src",
        command="python train.py --data_path ${{inputs.diabetes_data}}",
        inputs={
            "diabetes_data": Input(
                type="uri_file",
                path="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            )
        },
        environment=custom_env,
        compute="cluster-diabetes",
        experiment_name="exp-sdk-diabetes",
        display_name="treinamento-sdk-v2",
    )

    # 4. Submeter o Job
    print("Enviando o Job de Treinamento para o Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job enviado! Acompanhe em:\n{returned_job.studio_url}")

    # 5. Aguardar conclusão do Job (polling — evita problemas de streaming no CI)
    print("Aguardando conclusão do job...")
    import time
    while True:
        current_job = ml_client.jobs.get(returned_job.name)
        status = current_job.status
        print(f"  Status: {status}")
        if status in ("Completed", "Failed", "Canceled", "NotResponding"):
            break
        time.sleep(30)

    completed_job = ml_client.jobs.get(returned_job.name)
    print(f"Status final do job: {completed_job.status}")

    if completed_job.status != "Completed":
        print(f"Job encerrou com status '{completed_job.status}'. Abortando.")
        sys.exit(1)

    # 6. Ler métricas do job via MLflow
    print("\nLendo métricas do job...")
    try:
        tracking_uri = ml_client.workspaces.get(
            ml_client.workspace_name
        ).mlflow_tracking_uri
        mlflow_client = MlflowClient(tracking_uri=tracking_uri)

        # o run_id MLflow está nas properties do job
        run_id = completed_job.properties.get("mlflow.rootRunId") or returned_job.name
        print(f"  MLflow run_id: {run_id}")
        run = mlflow_client.get_run(run_id)
        metrics = run.data.metrics
        print(f"  Métricas encontradas: {list(metrics.keys())}")
    except Exception as e:
        print(f"Erro ao ler métricas: {e}")
        metrics = {}

    roc_auc  = metrics.get("roc_auc")
    accuracy = metrics.get("accuracy")

    if roc_auc is None or accuracy is None:
        print("ERRO: Métricas 'roc_auc' ou 'accuracy' não encontradas no run MLflow.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Acurácia : {accuracy:.4f}  (mínimo: {MIN_ACCURACY})")
    print(f"  AUC ROC  : {roc_auc:.4f}  (mínimo: {MIN_AUC_ROC})")
    print(f"{'='*50}")

    # 7. Gate de qualidade
    passed, reasons = evaluate_quality_gate(accuracy, roc_auc)

    if not passed:
        for reason in reasons:
            print(f"REPROVADO: {reason}")
        print("\nModelo não registrado. Melhore o modelo antes de promover.")
        sys.exit(1)

    print("\nModelo APROVADO. Registrando no Azure ML Model Registry...")

    # 8. Registrar o modelo
    model = Model(
        path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/",
        name="diabetes-rf-model",
        description="RandomForestClassifier treinado no dataset Pima Indians Diabetes",
        type=AssetTypes.MLFLOW_MODEL,
        tags={
            "framework": "scikit-learn",
            "algorithm": "RandomForestClassifier",
            "experiment": "exp-sdk-diabetes",
            "job_name": returned_job.name,
            "accuracy": str(round(accuracy, 4)),
            "roc_auc": str(round(roc_auc, 4)),
        },
    )

    registered_model = ml_client.models.create_or_update(model)

    print(f"\nModelo registrado com sucesso!")
    print(f"  Nome    : {registered_model.name}")
    print(f"  Versão  : {registered_model.version}")
    print(f"  AUC ROC : {roc_auc:.4f}")
    print(f"  Acurácia: {accuracy:.4f}")

    # 9. Criar ou atualizar o Batch Endpoint com a versão mais recente
    print("\n" + "="*50)
    print("Atualizando Batch Endpoint...")
    from create_batch_endpoint import create_or_update_batch_endpoint
    create_or_update_batch_endpoint(ml_client)
