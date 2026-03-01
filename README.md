# 🚀 AutoMLOps Studio

### Exploratory ML & MLOps Learning Engine

[![Version](https://img.shields.io/badge/Version-v1.2.0-blue)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

O **AutoMLOps Studio** é uma plataforma educacional "end-to-end" projetada para simplificar o ciclo de vida de Machine Learning. Desenvolvido de um **estudante para estudantes**, o projeto oferece uma interface intuitiva para explorar desde a ingestão de dados até o monitoramento de modelos em produção.

**🔗 Acesse a Demo ao vivo:** [Hugging Face Spaces - AutoMLOps Studio](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)

---

## 🎯 Objetivo e Problemática

Aprender MLOps muitas vezes exige lidar com infraestruturas complexas antes mesmo de entender os conceitos. Este projeto resolve isso ao centralizar:
- **Fluxo de Trabalho Unificado**: Uma jornada clara desde o upload de dados até o deploy.
- **Experimentação Visual**: Visualize o impacto de hiperparâmetros e arquiteturas em tempo real.
- **Conceitos de Produção**: Aprenda sobre Data Drift, Model Registry e Performance Monitoring sem precisar configurar servidores complexos.

## 👥 Público Alvo

- **Estudantes de Data Science**: Que buscam consolidar conhecimentos teóricos com prática visual.
- **Entusiastas de ML**: Que precisam de uma ferramenta rápida para prototipar modelos e testar hipóteses.
- **Desenvolvedores MLOps Junior**: Que desejam entender a integração entre ferramentas como MLflow, Optuna e APIs de predição.

---

## ✨ Funcionalidades e Detalhes Técnicos

### 1. 📊 Gestão de Dados & Drift Analysis (Aba Data)
- **Ingestão Inteligente**: Upload de CSVs com armazenamento versionado no Data Lake local.
- **Detecção de Drift Integrada**: Análise estatística (KS Test) e integração com **Deepchecks** para relatórios automáticos de integridade de dados (Data Integrity).

### 2. 🤖 Configuração de Treino (AutoML)
- **Suporte Multi-Tarefa**: Classificação, Regressão, Clustering, Séries Temporais e Detecção de Anomalias.
- **NLP & Transformers**: Pré-processamento otimizado para texto com suporte a TF-IDF, Count e Hugging Face Transformers (BERT, RoBERTa, etc.).
- **Otimização Avançada com Optuna**: Bayesian (TPE) com pruning precoce.
- **Ensemble Builder**: Criação dinâmica de Voting e Stacking Regressors/Classifiers.

### 3. 👁️ Visão Computacional (CV Engine)
- **Deep Learning Facilitado**: Classificação de Imagens, Segmentação Semântica (DeepLabV3) e Detecção de Objetos (Faster R-CNN).

### 4. ⚖️ Monitoring & Stability (Novo!)
- **Production Drift**: Dashboard para comparar dados de treinamento vs produção em tempo real.
- **Model Robustness & Stability**: Testes de estresse automáticos:
    - **Seed Stability**: Avalia a variação do modelo com diferentes inicializações.
    - **Split Stability**: Testa a robustez contra diferentes divisões dos dados (Data Variance).
    - **Hyperparameter Sensitivity**: Analisa o impacto de pequenas variações nos hiperparâmetros.

### 5. 🚀 Model Registry & Deployment
- **Gestão via MLflow**: Registro de modelos, controle de versões e estágios (Staging/Production).
- **Inference Playground**: Teste instantâneo de modelos registrados com suporte a colunas dinâmicas.
- **Self-Documenting Models**: Geração automática de relatórios detalhados contendo métricas e performance.


---

## 🧠 Aprendizados e Decisões Técnicas

### 1. Reorganização Centrada no Usuário
Aprendemos que separar "Drift" e "Monitoramento" em abas isoladas quebrava o raciocínio. Ao mover o **Drift para a aba de Dados** e o **Monitoramento para o Model Registry**, criamos um fluxo lógico:
- Primeiro você olha os dados (e o drift).
- Depois você treina e registra.
- Por fim, você faz o deploy e monitora.

### 2. Abstração de Infraestrutura
O uso de **Mock Deployments** permitiu ensinar o conceito de promoção de modelos e endpoints de API sem a necessidade de gerenciar clusters reais, tornando o aprendizado acessível a qualquer computador.

### 3. Otimização Inteligente
A implementação da **Validação Automática** ensina ao usuário que a estratégia de teste (CV vs Holdout) depende fundamentalmente do tamanho e tipo do dataset, uma lição crucial em Ciência de Dados.

---

## 📂 Estrutura do Projeto

- `app.py`: O coração do projeto - Dashboard Streamlit unificado.
- `automl_engine.py`: Motor de treinamento e otimização (Optuna/Scikit-Learn).
- `cv_engine.py`: Motor de Deep Learning para Visão Computacional (PyTorch/Torchvision).
- `mlops_utils.py`: Integração com MLflow e utilitários de sistema.
- `stability_engine.py`: Motor de testes de robustez e estabilidade.
- `api.py`: Backend FastAPI para servir modelos.
- `electron-main.js`: Wrapper para execução como aplicativo Desktop.

---

## 🚀 Como Executar

### 🐳 Via Docker (Recomendado)
```bash
docker-compose up --build
```
Acesse em `http://localhost:8501`.

### 🐍 Localmente (Python)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 🖥️ Desktop (Electron)
```bash
npm install
npm start
```

---

**Desenvolvido por Pedro Morato Lahoz.**
