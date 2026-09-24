import sys
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

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

# Verificar status final
completed_job = ml_client.jobs.get(returned_job.name)
print(f"Status final do job: {completed_job.status}")

if completed_job.status != "Completed":
    print(f"Job encerrou com status '{completed_job.status}'. Abortando registro do modelo.")
    sys.exit(1)

# 6. Registrar o modelo no Model Registry
print("Registrando o modelo no Azure ML Model Registry...")

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
    },
)

registered_model = ml_client.models.create_or_update(model)

print(f"Modelo registrado com sucesso!")
print(f"  Nome:    {registered_model.name}")
print(f"  Versão:  {registered_model.version}")
print(f"  ID:      {registered_model.id}")
