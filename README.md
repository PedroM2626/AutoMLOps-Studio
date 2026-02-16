# 🚀 AutoMLOps Studio
### Exploratory ML & MLOps Learning Engine

O **AutoMLOps Studio** é um projeto educacional desenvolvido de um **estudante para estudantes**. O objetivo principal é fornecer uma ferramenta prática para quem deseja explorar o mundo do Machine Learning ou criar modelos rapidamente para prototipagem e aprendizado. 

**Este projeto não é uma solução empresarial**, mas sim um laboratório interativo para aprender conceitos de AutoML, MLOps e Visão Computacional na prática, facilitando a experimentação rápida sem a necessidade de escrever centenas de linhas de código de infraestrutura.

## 🎯 Objetivo e Problemática
Muitas vezes, aprender Machine Learning parece fragmentado entre teoria e código complexo. Este projeto resolve isso ao centralizar:
- **Aprendizado Prático**: Entenda como o pré-processamento, o treinamento e o monitoramento se conectam.
- **Prototipagem Rápida**: Teste ideias de modelos em segundos com arquivos CSV ou imagens.
- **Desmistificação de MLOps**: Veja na prática como o versionamento de modelos (MLflow) e a detecção de desvios (Drift) funcionam em um fluxo real.

## 👥 Público Alvo
- **Estudantes de Ciência de Dados**: Que querem ver a teoria aplicada em uma interface visual.
- **Curiosos e Entusiastas de ML**: Que buscam uma ferramenta ágil para explorar datasets sem barreiras técnicas.
- **Desenvolvedores em Aprendizado**: Que desejam entender como integrar modelos de ML em APIs e Dashboards de forma simplificada.

## ✨ Funcionalidades

- **AutoML Tabular**: Suporte para Classificação, Regressão, Agrupamento (Clustering), Séries Temporais e Detecção de Anomalias com **Controle Total de Hiperparâmetros**.
- **Modelos Existentes (Fine-Tune)**: Aba integrada ao AutoML para carregar modelos do **Model Registry** ou arquivos locais para predição (Inference) ou retreinamento (Retraining) contra Data Drift.
- **Visualização Avançada**: Gráficos dinâmicos de performance, Matrizes de Confusão interativas, Curvas Real vs Predito e **Projeções PCA** para visualização de clusters e anomalias.
- **MLOps Completo**: Integração profunda com **MLflow 3.9.0** para rastreamento automático de **todos** os experimentos. Salva automaticamente:
    - Tipo do modelo e configurações.
    - Todos os hiperparâmetros utilizados.
    - Métricas de avaliação detalhadas.
    - Artefatos (modelos serializados, gráficos, logs).
- **Computer Vision**: Fine-tuning de modelos para Classificação e **Segmentação Semântica** (DeepLabV3).
- **Amplo Suporte a Modelos**: 
    - **Classificação/Regressão**: RandomForest, XGBoost, LightGBM, SVM (SVC/SVR/LinearSVC/LinearSVR), KNN, Naive Bayes, MLP (Neural Networks), Ridge, Lasso, ElasticNet, Logistic Regression, Decision Tree, Gradient Boosting, AdaBoost, CatBoost.
    - **Clustering**: K-Means, DBSCAN, Agglomerative Clustering, Spectral Clustering, Gaussian Mixture.
    - **Anomaly Detection**: Isolation Forest, Local Outlier Factor, One-Class SVM.
- **Estratégias de Split Inteligentes**: Split aleatório e **Split Temporal** automático para séries temporais.
- **Explicabilidade (SHAP)**: Integração com SHAP para entender a importância das features em modelos de classificação.
- **🐳 Docker Ready**: Orquestração multi-serviço (API, Dashboard, MLflow) pronta para deploy.
- **🔌 REST API**: Camada de serving baseada em FastAPI com autenticação via API Key.

## 📂 Estrutura do Projeto

- `app.py`: Dashboard interativo em Streamlit.
- `automl_engine.py`: Core de pré-processamento, treinamento e otimização.
- `cv_engine.py`: Motor para tarefas de Visão Computacional.
- `mlops_utils.py`: Utilitários de MLOps (MLflow, Data Lake, Drift, SHAP).
- `api.py`: API de serving de modelos.
- `docker-compose.yml` & `Dockerfile`: Configurações de containerização.
- `tests/`: Suíte de testes automatizados.

## 🚀 Como Começar

### Via Docker (Recomendado)

A forma mais rápida de rodar toda a stack (Dashboard, API e MLflow):

```bash
docker-compose up --build
```

- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000

### Instalação Local

1. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

2. **Execute o Dashboard (Streamlit)**:
```bash
python -m streamlit run app.py
```

3. **Execute a API**:
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## 🛠️ Guia de Uso do Dashboard

1.  **📊 Data**: Faça o upload do seu CSV e salve no **Data Lake** para habilitar o versionamento.
2.  **🤖 AutoML**: 
    - **Novo Treino**: Configure o treino automático ou manual. Selecione modelos, defina a estratégia de hiperparâmetros e acompanhe o progresso em tempo real.
    - **Modelos Existentes (Fine-Tune)**: Gerencie modelos já treinados. Carregue do Registry ou via upload para prever novos dados ou retreinar o modelo com dados atualizados do Data Lake.
3.  **🧪 Experiments**: Explore o histórico de treinos, compare métricas e registre os melhores modelos.
4.  **🖼️ Computer Vision**: Treine modelos de classificação de imagens.
5.  **📈 Drift/Monitoring**: Detecte desvios estatísticos entre dados de referência e atuais.
6.  **🗂️ Model Registry**: Catálogo oficial de modelos aprovados para produção.

## 🧪 Testes

A plataforma inclui uma suíte completa de testes automatizados para garantir a qualidade e a integração dos componentes.

### Executando os Testes

Para rodar todos os testes do projeto:

```bash
pytest tests/
```

### Principais Testes Incluídos:

- **Integração MLflow (`tests/test_mlflow_integration.py`)**: Verifica se os experimentos, parâmetros e métricas são corretamente registrados no MLflow.
- **Fluxo AutoML (`tests/test_automl_tab.py`)**: Simula o pipeline completo de treinamento para classificação e regressão via interface.
- **Simulação de Interface (`tests/test_interface_simulation_unified.py`)**: Valida a interação dos componentes da UI com o motor de AutoML.
- **Transformers (`tests/test_automl_transformers.py`)**: Testa a integração (mockada) com modelos de NLP da Hugging Face.
- **Reprodutibilidade (`tests/test_reproducibility.py`)**: Garante que os resultados sejam consistentes entre execuções.

## 📦 Dependências e Ambiente

O projeto utiliza um arquivo `requirements.txt` com versões pinadas para garantir a estabilidade. As principais dependências incluem:

- **MLflow 3.9.0**: Para rastreamento de experimentos e registro de modelos.
- **Streamlit**: Para a interface do dashboard.
- **FastAPI**: Para a API de serving.
- **Scikit-learn, XGBoost, LightGBM**: Motores de machine learning.

### Docker (Recomendado)

O ambiente é totalmente containerizado. O `Dockerfile` utiliza `python:3.11-slim` para uma imagem leve e eficiente.

```bash
docker-compose up --build
```

Isso iniciará:
- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000 (Versão 3.9.0)

## 🛠️ Configuração

Configure as variáveis de ambiente no arquivo `.env`:
- `API_SECRET_KEY`: Chave de segurança para a API REST.
- `MLFLOW_TRACKING_URI`: Localização dos logs do MLflow (padrão: `./mlruns`).

---
Desenvolvido por Pedro Morato Lahoz
