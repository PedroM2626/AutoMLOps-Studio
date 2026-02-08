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

- **AutoML Tabular**: Suporte para Classificação, Regressão, Agrupamento (Clustering), Séries Temporais e Detecção de Anomalias com **Hiperparâmetros Automáticos ou Manuais**.
- **Modelos Existentes (Fine-Tune)**: Aba integrada ao AutoML para carregar modelos do **Model Registry** ou arquivos locais para predição (Inference) ou retreinamento (Retraining) contra Data Drift.
- **Visualização Avançada**: Gráficos dinâmicos de performance, Matrizes de Confusão interativas, Curvas Real vs Predito e **Projeções PCA** para visualização de clusters e anomalias.
- **Computer Vision**: Fine-tuning de modelos para Classificação e **Segmentação Semântica** (DeepLabV3).
- **Modelos Expandidos**: Inclui RandomForest, XGBoost, LightGBM, SVM, LinearSVC, KNN, Naive Bayes, MLP, Ridge, Lasso, ElasticNet, e muito mais.
- **Estratégias de Split Inteligentes**: Split aleatório e **Split Temporal** automático para séries temporais.
- **Explicabilidade (SHAP)**: Integração com SHAP para entender a importância das features em modelos de classificação.
- **🐳 Docker Ready**: Orquestração multi-serviço (API, Dashboard, MLflow) pronta para deploy.
- **🔌 REST API**: Camada de serving baseada em FastAPI com autenticação via API Key.

## 📂 Estrutura do Projeto

- `app.py`: Dashboard interativo em Streamlit.
- `flet_app.py`: Versão cross-platform (Desktop/Mobile/Web) baseada em Flet.
- `simple_flet_app.py`: Interface simples de teste com Flet para verificação rápida do ambiente.
- `automl_engine.py`: Core de pré-processamento, treinamento e otimização.
- `cv_engine.py`: Motor para tarefas de Visão Computacional.
- `mlops_utils.py`: Utilitários de MLOps (MLflow, Data Lake, Drift, SHAP).
- `api.py`: API de serving de modelos.
- `docker-compose.yml` & `Dockerfile`: Configurações de containerização.
- `tests.py`: Suíte de testes unitários, integração e aceitação.

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

3. **Execute o Dashboard Modular (Flet)**:
```bash
python flet_app/src/main.py
```

4. **Execute a API**:
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## 🏗️ Arquitetura do Flet App (Modular)

O novo Dashboard em Flet segue uma arquitetura modularizada inspirada no `gallery-main`, facilitando a manutenção e escalabilidade:

- **`flet_app/src/main.py`**: Ponto de entrada que inicializa os contextos e a estrutura principal.
- **`flet_app/src/contexts/`**: Provedores de estado global (Tema, Roteamento).
- **`flet_app/src/components/`**: Componentes reutilizáveis (AppBar, Navigation).
- **`flet_app/src/views/`**: Telas individuais da aplicação (Data, Train, CV, Experiments, Registry).
- **`flet_app/src/models/`**: Gerenciamento de estado centralizado (`app_state.py`).

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

A plataforma inclui uma suíte completa de testes:
```bash
# Testes do Core
pytest tests.py

# Testes da Interface Flet
pytest tests_flet_app.py
pytest tests_acceptance_flet.py
```
- **Unitários**: Processamento de dados, instanciação de modelos e lógica de interface.
- **Integração**: Salvamento de pipelines, utilitários de MLOps e carregamento de componentes UI.
- **Aceitação**: Fluxos completos de treino simulados e interação via browser (Playwright) para a interface Flet.

## 🛠️ Configuração

Configure as variáveis de ambiente no arquivo `.env`:
- `API_SECRET_KEY`: Chave de segurança para a API REST.
- `MLFLOW_TRACKING_URI`: Localização dos logs do MLflow (padrão: `./mlruns`).

---
Desenvolvido por Pedro Morato Lahoz
