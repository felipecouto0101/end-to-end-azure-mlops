from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
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
    compute="cluster-diabetes",  # Nome do seu cluster de cálculo existente
    experiment_name="exp-sdk-diabetes",
    display_name="treinamento-sdk-v2",
)

# 4. Submeter o Job para a nuvem Azure ML
print("Enviando o Job de Treinamento para o Azure ML...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job enviado com sucesso! Acompanhe a execução pela URL:\n{returned_job.studio_url}")
