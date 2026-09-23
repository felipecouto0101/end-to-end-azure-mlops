# 🩺 End-to-End Azure Machine Learning & MLOps: Diabetes Prediction

Este repositório contém o ciclo completo de implementação de um modelo de Machine Learning no **Azure Machine Learning Studio**, cobrindo desde a experimentação visual até a automação de treinos usando **SDK v2**, ambientes isolados via Conda/Docker e boas práticas de MLOps.

---

## 📌 Visão Geral do Projeto

O objetivo principal é construir um modelo preditivo para identificar a probabilidade de um paciente desenvolver diabetes com base em métricas clínicas (níveis de glicose, IMC, idade, pressão arterial, entre outras).

### Abordagens Implementadas

1. **Projeto 1 — Pipeline Visual (Azure ML Designer):** Fluxo no-code/low-code utilizando o Azure ML Designer para pré-processamento, divisão de dados, treinamento com *Two-Class Decision Forest* e avaliação de desempenho.
2. **Projeto 2 — MLOps & Automação (Azure ML SDK v2):** Submissão automatizada de jobs de treinamento em clusters de computação dedicados, rastreabilidade via Git e controle de versão de ambientes.

---

## 🛠️ Arquitetura e Tecnologias

| Componente | Tecnologia |
|---|---|
| Nuvem | Microsoft Azure ML |
| Linguagem | Python 3.10 |
| Rastreamento | MLflow + Azure ML autolog |
| Computação | Azure ML Compute Cluster (`cluster-diabetes`) |
| Ambiente | Conda + Docker (`diabetes-ml-env`) |
| Orquestração | Azure ML SDK v2 (`azure-ai-ml`) |

---

## 🚀 Estrutura do Pipeline de ML

```text
[Dataset: Pima Indians Diabetes]
             │
             ▼
      [Split Data (80/20)]
       │              │
       ▼              ▼
[Train Model]   [Score Model]
       │              │
       └──────┬───────┘
              ▼
       [Evaluate Model]
    (Acurácia, AUC ROC)
```

### Módulos do Pipeline Visual

- **Data Input:** Leitura do arquivo `.csv` registrado no Azure Data Assets.
- **Split Data:** Divisão estratificada em treino e teste.
- **Algorithm:** Classificação binária (*Two-Class Decision Forest*).
- **Train Model:** Treinamento mapeando a variável alvo (`Outcome`).
- **Score & Evaluate Model:** Métricas de desempenho — Acurácia, Precisão, Recall, Curva ROC/AUC.

---

## 📁 Estrutura do Repositório

```bash
end-to-end-azure-mlops/
├── azureml/
│   ├── conda.yml          # Dependências do ambiente Python (scikit-learn, mlflow, etc.)
│   └── run_pipeline.py    # Orquestrador: conecta ao Workspace e submete o job via SDK v2
├── src/
│   └── train.py           # Script de treinamento do modelo (RandomForest + MLflow autolog)
└── README.md
```

---

## 💻 Como Executar via SDK v2

### Pré-requisitos

- Assinatura ativa do Microsoft Azure
- Workspace do Azure Machine Learning configurado
- Python 3.8+ com `pip`

### Passo a Passo

**1. Clonar o repositório:**
```bash
git clone https://github.com/felipecouto0101/end-to-end-azure-mlops.git
cd end-to-end-azure-mlops
```

**2. Instalar as dependências de orquestração:**
```bash
pip install azure-ai-ml azure-identity mlflow azureml-mlflow
```

**3. Submeter o job de treinamento:**
```bash
python azureml/run_pipeline.py
```

**4. Acompanhar a execução:**

Acesse a aba **Jobs** no [Azure Machine Learning Studio](https://ml.azure.com) para visualizar métricas do MLflow, logs e artefatos gerados em tempo real.

---

## 📊 Boas Práticas de MLOps Aplicadas

- [x] **Infraestrutura como Código:** Computação gerenciada e sob demanda via SDK v2
- [x] **Reprodutibilidade:** Ambientes isolados via containers Conda/Docker versionados
- [x] **Rastreabilidade:** Commits do Git vinculados automaticamente às execuções no Azure ML
- [x] **Gestão de Custos:** Cluster configurado com auto-scale (escala para 0 nós quando ocioso)
- [x] **Logging Automático:** MLflow `autolog` registra hiperparâmetros, métricas e artefatos sem código adicional

---

## ✒️ Autor

Desenvolvido por **Felipe Couto** para fins de aprendizado e consolidação de conhecimentos na plataforma Microsoft Azure ML.
