# 🚀 AutoMLOps Studio

### Exploratory ML & MLOps Learning Engine

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)

O **AutoMLOps Studio** é um projeto educacional desenvolvido de um **estudante para estudantes**. O objetivo principal é fornecer uma ferramenta prática para quem deseja explorar o mundo do Machine Learning ou criar modelos rapidamente para prototipagem e aprendizado.

**Este projeto não é uma solução empresarial**, mas sim um laboratório interativo para aprender conceitos de AutoML, MLOps e Visão Computacional na prática, facilitando a experimentação rápida sem a necessidade de escrever centenas de linhas de código de infraestrutura.

## 🎯 Objetivo e Problemática

Muitas vezes, aprender Machine Learning parece fragmentado entre teoria e código complexo. Este projeto resolve isso ao centralizar:
- **Aprendizado Prático**: Entenda como o pré-processamento, o treinamento e o monitoramento se conectam.
- **Prototipagem Rápida**: Teste ideias de modelos em segundos com arquivos CSV ou imagens.
- **Desmistificação de MLOps**: Veja na prática como o versionamento de modelos (MLflow), integração com DagsHub e a detecção de desvios (Drift) funcionam em um fluxo real.

## 👥 Público Alvo

- **Estudantes de Ciência de Dados**: Que querem ver a teoria aplicada em uma interface visual.
- **Curiosos e Entusiastas de ML**: Que buscam uma ferramenta ágil para explorar datasets sem barreiras técnicas.
- **Desenvolvedores em Aprendizado**: Que desejam entender como integrar modelos de ML em APIs e Dashboards de forma simplificada.

## ✨ Funcionalidades Principais

### 🧠 Otimização de Hiperparâmetros Avançada
Agora você pode escolher **manualmente** a estratégia de otimização que melhor se adapta ao seu problema:
- **Bayesian Optimization (Padrão)**: Utiliza Processos Gaussianos (TPE) para encontrar os melhores hiperparâmetros de forma eficiente.
- **Random Search**: Exploração aleatória do espaço de busca, ideal para benchmarks.
- **Grid Search**: Busca exaustiva (fallback para Random se o espaço for dinâmico).
- **Hyperband**: Método avançado que descarta configurações ruins rapidamente (Bandit-based), ideal para grandes volumes de dados.

### 🤖 Validação Automática Inteligente
O sistema agora conta com um modo **Automático** para escolha da estratégia de validação:
- **Séries Temporais**: Detecta automaticamente e aplica `TimeSeriesSplit`.
- **Pequenos Datasets (<1000 amostras)**: Aplica `Cross-Validation` para garantir robustez.
- **Grandes Datasets (>=1000 amostras)**: Aplica `Holdout (Train-Test Split)` para eficiência.

### 📊 Outras Funcionalidades
- **AutoML Tabular**: Classificação, Regressão, Clustering, Séries Temporais, Detecção de Anomalias.
- **Performance**: Paralelismo total (`n_jobs=-1`) e integração com Optuna.
- **Integração MLOps**: Rastreamento completo via MLflow (parâmetros, métricas, artefatos).
- **Explicabilidade**: Integração nativa com SHAP.
- **Data Lake**: Versionamento de datasets brutos e processados.
- **Docker Ready**: Ambiente containerizado pronto para uso.

## 📂 Estrutura do Projeto

- `app.py`: Interface principal (Streamlit Dashboard).
- `automl_engine.py`: Motor de AutoML (Treinamento, Otimização, Validação).
- `cv_engine.py`: Motor de Visão Computacional.
- `mlops_utils.py`: Utilitários de MLOps.
- `api.py`: API de serving (FastAPI).
- `test_interface_simulation.py`: Script de teste para validação das funcionalidades de otimização e interface.
- `docker-compose.yml` & `Dockerfile`: Configuração de containers.

## 🚀 Instalação e Uso

### Pré-requisitos
- Docker e Docker Compose instalados.
- (Opcional) Python 3.11+ para execução local.

### 🐳 Via Docker (Recomendado)

1. **Clone o repositório**:
   ```bash
   git clone <url-do-repositorio>
   cd automlops-studio
   ```

2. **Configure as variáveis de ambiente**:
   Copie o exemplo e ajuste conforme necessário (opcional para rodar localmente):
   ```bash
   cp .env.example .env
   ```

3. **Suba os containers**:
   ```bash
   docker-compose up --build
   ```

4. **Acesse os serviços**:
   - **Dashboard (Streamlit)**: [http://localhost:8501](http://localhost:8501)
   - **API (FastAPI)**: [http://localhost:8000](http://localhost:8000)
   - **MLflow UI**: [http://localhost:5000](http://localhost:5000)

### 🐍 Execução Local (Sem Docker)

1. **Crie um ambiente virtual e instale dependências**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

2. **Execute o Dashboard**:
   ```bash
   python -m streamlit run app.py
   ```

3. **(Opcional) Execute a API**:
   ```bash
   python -m uvicorn api:app --reload
   ```

## 🧪 Testes e Validação

Para verificar se todas as funcionalidades de otimização (Grid, Random, Bayesian, Hyperband) e validação automática estão funcionando corretamente, execute o script de simulação:

```bash
python test_interface_simulation.py
```
Este script simula o comportamento da interface utilizando os datasets disponíveis no `data_lake`.

---
**Desenvolvido por Pedro Morato Lahoz.**
