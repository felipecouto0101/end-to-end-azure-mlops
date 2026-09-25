import sys
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# ── Limiares de qualidade para promoção do modelo ─────────────────────────────
MIN_AUC_ROC  = 0.82   # métrica principal — dataset desbalanceado
MIN_ACCURACY = 0.74   # métrica secundária

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

# 5. Aguardar conclusão do Job
print("Aguardando conclusão do job...")
ml_client.jobs.stream(returned_job.name)

completed_job = ml_client.jobs.get(returned_job.name)
print(f"Status final do job: {completed_job.status}")

if completed_job.status != "Completed":
    print(f"Job encerrou com status '{completed_job.status}'. Abortando.")
    sys.exit(1)

# 6. Ler métricas do job via MLflow
print("\nLendo métricas do job...")
from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient(
    tracking_uri=ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
)

run = mlflow_client.get_run(returned_job.name)
metrics = run.data.metrics

roc_auc  = metrics.get("roc_auc")
accuracy = metrics.get("accuracy")

if roc_auc is None or accuracy is None:
    print("ERRO: Métricas 'roc_auc' ou 'accuracy' não encontradas no run MLflow.")
    sys.exit(1)

print(f"\n{'='*50}")
print(f"  Acurácia : {accuracy:.4f}  (mínimo: {MIN_ACCURACY})")
print(f"  AUC ROC  : {roc_auc:.4f}  (mínimo: {MIN_AUC_ROC})")
print(f"{'='*50}")

# 7. Gate de qualidade — só promove se ambas as métricas passarem
passed = True

if roc_auc < MIN_AUC_ROC:
    print(f"REPROVADO: AUC ROC {roc_auc:.4f} abaixo do limiar {MIN_AUC_ROC}")
    passed = False

if accuracy < MIN_ACCURACY:
    print(f"REPROVADO: Acurácia {accuracy:.4f} abaixo do limiar {MIN_ACCURACY}")
    passed = False

if not passed:
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
