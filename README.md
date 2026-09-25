# 🩺 End-to-End Azure Machine Learning & MLOps: Diabetes Prediction

Este repositório contém o ciclo completo de implementação de um modelo de Machine Learning no **Azure Machine Learning Studio**, cobrindo desde a experimentação visual até a automação de treinos com **CI/CD via GitHub Actions**, testes automatizados e boas práticas de MLOps.

---

## 📌 Visão Geral do Projeto

O objetivo é construir um modelo preditivo para identificar a probabilidade de um paciente desenvolver diabetes com base em métricas clínicas (níveis de glicose, IMC, idade, pressão arterial, entre outras).

### Abordagens Implementadas

1. **Projeto 1 — Pipeline Visual (Azure ML Designer):** Fluxo no-code/low-code com pré-processamento, divisão de dados, treinamento com *Two-Class Decision Forest* e avaliação de desempenho.
2. **Projeto 2 — MLOps & Automação (Azure ML SDK v2):** Submissão automatizada de jobs de treinamento em clusters dedicados, rastreabilidade via Git, controle de versão de ambientes e CI/CD completo.

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
| CI/CD | GitHub Actions |
| Testes unitários | pytest |
| Qualidade de dados | Great Expectations 1.x |

---

## 🔄 Pipeline CI/CD

O workflow do GitHub Actions está dividido em dois jobs sequenciais:

```
git push / PR
     │
     ▼
Job 1: test
     ├── pytest tests/test_train.py        (14 testes unitários)
     ├── pytest tests/test_data_quality.py (13 validações de dados)
     └── pytest tests/test_quality_gate.py (11 testes do gate de qualidade)
          │
          ├── falhou? → pipeline bloqueado, treinamento não roda
          └── passou? ↓
     ▼
Job 2: train  (só dispara com tag v*)
     ├── az login (Service Principal)
     ├── python azureml/run_pipeline.py
     │        ├── submete job ao cluster-diabetes
     │        ├── aguarda conclusão
     │        ├── lê métricas via MLflow
     │        ├── gate de qualidade (AUC ROC >= 0.82 e Acurácia >= 0.74)
     │        │        ├── reprovado? → pipeline falha, modelo não registado
     │        │        └── aprovado? ↓
     │        └── registra modelo no Azure ML Model Registry
     └── modelo versionado disponível no Azure ML Studio
```

### Como disparar o treinamento

```bash
# push normal — roda apenas os testes (sem custo Azure)
git push

# release — roda testes + treinamento no Azure ML
git tag v1.0.0
git push origin v1.0.0
```

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
    (Acurácia: 76.6% | AUC ROC: 83.5%)
```

---

## 📁 Estrutura do Repositório

```bash
end-to-end-azure-mlops/
├── .github/
│   └── workflows/
│       └── train.yml          # CI/CD: testes + treinamento no Azure ML
├── azureml/
│   ├── conda.yml              # Dependências do ambiente Python no cluster
│   └── run_pipeline.py        # Orquestrador: submete job e registra modelo via SDK v2
├── src/
│   └── train.py               # Script de treinamento modular (RandomForest + MLflow)
├── tests/
│   ├── test_train.py          # Testes unitários das funções de treino (pytest)
│   ├── test_data_quality.py   # Validações do dataset (Great Expectations)
│   └── test_quality_gate.py   # Testes do gate de qualidade (aprovação/reprovação do modelo)
├── requirements-dev.txt       # Dependências de teste
└── README.md
```

---

## 🧪 Testes Automatizados

### Testes Unitários — `pytest`

Cobrem todas as funções do `train.py` de forma isolada, sem necessidade de Azure ou dados reais:

| Função | O que é testado |
|---|---|
| `load_data()` | retorna DataFrame, colunas corretas, não vazio |
| `split_data()` | 4 outputs, proporção 80/20, Outcome ausente nas features |
| `train_model()` | tipo RandomForestClassifier, hiperparâmetros, modelo treinado |
| `evaluate_model()` | dict com accuracy e roc_auc, valores entre 0 e 1 |

```bash
pytest tests/test_train.py -v
# 14 passed
```

### Validação de Dados — `Great Expectations`

Valida o dataset antes do treinamento com 13 regras:

- Todas as 9 colunas presentes
- Mínimo de 500 linhas
- Sem nulos nas colunas críticas (Glucose, BMI, Age, Outcome)
- `Outcome` contém apenas 0 e 1 (ambas as classes presentes)
- Faixas clínicas realistas (Glucose: 0-300, BMI: 10-70, Age: 1-120)
- Todos os campos com tipos numéricos

```bash
pytest tests/test_data_quality.py -v
# 13 passed
```

### Gate de Qualidade — `evaluate_quality_gate`

Bloqueia o registo do modelo no Azure ML se as métricas não atingirem os limiares mínimos:

| Métrica | Limiar | Justificativa |
|---|---|---|
| AUC ROC | `>= 0.82` | Métrica principal — dataset desbalanceado (65%/35%) |
| Acurácia | `>= 0.74` | Significativamente acima do baseline (65%) |

```bash
pytest tests/test_quality_gate.py -v
# 11 passed
```

---

## 💻 Como Executar Localmente

### Pré-requisitos

- Assinatura ativa do Microsoft Azure com Workspace configurado
- Python 3.10+

### Passo a Passo

**1. Clonar o repositório:**
```bash
git clone https://github.com/felipecouto0101/end-to-end-azure-mlops.git
cd end-to-end-azure-mlops
```

**2. Instalar dependências de orquestração:**
```bash
pip install azure-ai-ml azure-identity mlflow azureml-mlflow
```

**3. Instalar dependências de teste:**
```bash
pip install -r requirements-dev.txt
```

**4. Rodar os testes:**
```bash
pytest tests/ -v
```

**5. Submeter o job de treinamento:**
```bash
python azureml/run_pipeline.py
```

**6. Acompanhar a execução:**

Acesse a aba **Jobs** no [Azure Machine Learning Studio](https://ml.azure.com) para visualizar métricas do MLflow, logs e artefatos em tempo real.

---

## 📊 Boas Práticas de MLOps Aplicadas

- [x] **Infraestrutura como Código:** Computação gerenciada e sob demanda via SDK v2
- [x] **Reprodutibilidade:** Ambientes isolados via containers Conda/Docker versionados
- [x] **Rastreabilidade:** Commits do Git vinculados automaticamente às execuções no Azure ML
- [x] **Gestão de Custos:** Cluster com auto-scale (escala para 0 nós quando ocioso); treinamento só dispara em tags, não em todo push
- [x] **Logging Automático:** MLflow `autolog` registra hiperparâmetros, métricas e artefatos
- [x] **Testes Automatizados:** Validação do código, dos dados e do gate de qualidade antes de qualquer treinamento
- [x] **Model Registry:** Modelo versionado e registado automaticamente no Azure ML após cada release, desde que passe no gate de qualidade (AUC ROC >= 0.82, Acurácia >= 0.74)


