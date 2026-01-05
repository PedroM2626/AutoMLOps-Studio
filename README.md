# Free MLOps Platform 🚀

Plataforma **Enterprise-Grade**, **100% gratuita** e **self-hosted** para criar, treinar, avaliar e fazer deploy de modelos de Machine Learning com recursos avançados comparáveis às principais plataformas comerciais.

## 🎯 Status da Plataforma: **PRODUCTION-READY**

### ✅ Funcionalidades Implementadas (9/20)

#### 📊 **Model Monitoring & Observability** ✅
- **Performance Monitoring:** accuracy, latency, throughput em tempo real
- **Data Drift Detection:** análise estatística de mudanças nos dados
- **Concept Drift Detection:** detecção de mudanças no comportamento do modelo
- **Alert System:** alertas automáticos baseados em thresholds configuráveis

#### 🔬 **Experiment Management Avançado** ✅
- **Hyperparameter Optimization:** otimização com Optuna (TPE Sampler)
- **Cross-validation:** validação cruzada configurável
- **Study Management:** histórico completo de otimizações
- **Best Model Selection:** seleção automática dos melhores hiperparâmetros

#### 📦 **Data Versioning & Lineage** ✅
- **DVC Integration:** controle de versão de datasets e pipelines
- **Data Lineage:** rastreabilidade completa de upstream/downstream
- **Data Validation:** validação automática com Pandera schemas
- **Schema Management:** criação, comparação e exportação de schemas

#### 🤖 **Extended AutoML** ✅
- **Deep Learning:** TensorFlow e PyTorch (MLP, CNN, LSTM)
- **Advanced DL:** TabTransformer, Vision Transformer, Attention mechanisms
- **Time Series:** ARIMA, Prophet, LSTM para séries temporais
- **Multi-framework:** suporte a múltiplos frameworks de ML/DL
- **Auto-tuning:** configuração automática de hiperparâmetros
- **MLflow Tracking:** experiment tracking profissional
- **Model Explainability:** SHAP e Captum para interpretabilidade

---

## 🏆 Comparação com Plataformas Comerciais

| Funcionalidade | Free MLOps | Azure ML | SageMaker | Vertex AI | Databricks |
|---|---|---|---|---|---|
| **AutoML Clássico** | ✅ Scikit-learn | ✅ | ✅ | ✅ | ✅ |
| **Deep Learning** | ✅ TF/PyTorch | ✅ | ✅ | ✅ | ✅ |
| **Advanced DL** | ✅ Transformers | ❌ | ❌ | ✅ | ✅ |
| **Time Series** | ✅ ARIMA/Prophet/LSTM | ✅ | ✅ | ✅ | ✅ |
| **Hyperparameter Opt** | ✅ Optuna | ✅ | ✅ | ✅ | ✅ |
| **Model Monitoring** | ✅ Performance/Drift | ✅ | ✅ | ✅ | ✅ |
| **Data Versioning** | ✅ DVC | ✅ | ✅ | ✅ | ✅ |
| **Data Validation** | ✅ Pandera | ✅ | ✅ | ✅ | ✅ |
| **Experiment Tracking** | ✅ SQLite | ✅ | ✅ | ✅ | ✅ |
| **Model Registry** | ✅ Versionamento | ✅ | ✅ | ✅ | ✅ |
| **API Deployment** | ✅ FastAPI | ✅ | ✅ | ✅ | ✅ |
| **Custo** | **GRATIS** | $$$$ | $$$$ | $$$$ | $$$$ |
| **Self-Hosted** | ✅ | ❌ | ❌ | ❌ | ❌ |

**🎉 Free MLOps oferece recursos enterprise-grade com 100% controle dos dados e zero custo!**

---

## 🚀 Funcionalidades Principais

### 📈 **Core MLOps**
- **AutoML Clássico:** 13 algoritmos scikit-learn com tuning automático
- **Experiment Tracking:** SQLite com versionamento completo
- **Model Registry:** registro e versionamento de modelos
- **Fine-Tuning:** GridSearchCV e RandomizedSearchCV
- **Model Testing:** testes individuais e em lote

### 🤖 **Advanced AutoML**
- **Deep Learning:** MLP, CNN, LSTM com TensorFlow/PyTorch
- **Time Series:** ARIMA, Prophet, LSTM para forecasting
- **Hyperparameter Optimization:** Optuna com TPE Sampler
- **Neural Architecture Search:** planejado para implementação

### 📊 **Enterprise Monitoring**
- **Real-time Performance:** accuracy, latency, throughput
- **Drift Detection:** data drift e concept drift
- **Alert System:** thresholds configuráveis
- **Dashboard Completo:** visualizações interativas

### 📦 **Data Management**
- **DVC Integration:** versionamento de datasets e pipelines
- **Data Validation:** schemas Pandera com validação automática
- **Data Lineage:** rastreabilidade completa
- **Schema Management:** criação e comparação de schemas

---

## 🛠️ Instalação Completa

### Requisitos Base
- Python **3.10+**

### Instalação Básica
```bash
pip install -r requirements.txt
```

### Instalação Completa (todas as funcionalidades)
```bash
pip install -r requirements-full.txt
```

### Instalação Flexível (recomendado)

#### 🎯 **Opção 1: Instalação Guiada**
```bash
# Instalar dependências essenciais
pip install -r requirements.txt

# Instalar dependências opcionais conforme necessidade
python install_optional.py --all                    # Todas as funcionalidades
python install_optional.py --deep-learning          # Apenas Deep Learning
python install_optional.py --time-series            # Apenas Time Series
python install_optional.py --monitoring             # Apenas Monitoring
python install_optional.py --data-validation         # Apenas Data Validation
```

#### 🎯 **Opção 2: Arquivos de Requirements**
```bash
# Mínimo para funcionamento básico
pip install -r requirements-base.txt

# Funcionalidades básicas + visualizações
pip install -r requirements.txt

# Todas as funcionalidades (produção completa)
pip install -r requirements-full.txt
```

#### 🎯 **Opção 3: Manual por Funcionalidade**
```bash
# 🤖 Deep Learning
pip install tensorflow torch

# 🚀 Advanced Deep Learning (Transformers)
pip install transformers mlflow shap captum

# 📈 Time Series  
pip install statsmodels pmdarima prophet

# 🔬 Hyperparameter Optimization
pip install optuna

# 📦 Data Versioning & Validation
pip install dvc pandera

# 📊 Visualizações Avançadas
pip install plotly
```

---

## 🎮 Como Usar

### Interface Web (Recomendado)
```bash
streamlit run free_mlops/streamlit_app.py
```

Acesse: `http://localhost:8501`

### 🚀 MLflow Tracking (Opcional)
```bash
# Iniciar MLflow UI para experiment tracking
python start_mlflow.py
```

Acesse: `http://localhost:5000`

### 🐳 Docker (Recomendado para produção)
```bash
# Build e iniciar todos os serviços
./docker-run.sh run

# Ou individualmente
./docker-run.sh streamlit    # Apenas Streamlit
./docker-run.sh mlflow       # Apenas MLflow
./docker-run.sh api          # Apenas API

# Parar serviços
./docker-run.sh stop

# Ver logs
./docker-run.sh logs

# Limpar tudo
./docker-run.sh cleanup
```

**Acessar serviços:**
- 🌐 Streamlit: http://localhost:8501
- 📊 MLflow UI: http://localhost:5000
- 🔌 API Docs: http://localhost:8000/docs

### ☸️ Kubernetes (Para clusters)
```bash
# Deploy completo
./k8s-deploy.sh deploy

# Com ingress
./k8s-deploy.sh deploy --ingress

# Ver status
./k8s-deploy.sh status

# Ver logs
./k8s-deploy.sh logs free-mlops-app

# Escalar
./k8s-deploy.sh scale free-mlops-app 3

# Atualizar
./k8s-deploy.sh update

# Remover
./k8s-deploy.sh delete
```

### API REST
```bash
python -m free_mlops.api
```

Acesse: `http://localhost:8000/docs`

---

## 📋 Estrutura do Projeto

```text
free-mlops/
├── free_mlops/
│   ├── 🎯 Core MLOps
│   │   ├── automl.py              # AutoML clássico
│   │   ├── service.py             # Serviços principais
│   │   ├── db.py                  # Banco de dados
│   │   ├── registry.py            # Model Registry
│   │   ├── finetune.py            # Fine-tuning
│   │   └── test_models.py         # Teste de modelos
│   │
│   ├── 📊 Monitoring & Observability
│   │   ├── monitoring.py          # Performance monitoring
│   │   ├── drift_detection.py     # Data drift detection
│   │   ├── concept_drift.py       # Concept drift detection
│   │   └── alert_manager.py       # Sistema de alertas
│   │
│   ├── 🔬 Advanced Experiment Management
│   │   └── hyperopt.py            # Hyperparameter optimization
│   │
│   ├── 📦 Data Versioning & Lineage
│   │   ├── dvc_integration.py     # DVC integration
│   │   └── data_validation.py     # Data validation com Pandera
│   │
│   ├── 🤖 Extended AutoML
│   │   ├── deep_learning.py       # TensorFlow/PyTorch (MLP, CNN, LSTM)
│   │   ├── advanced_deep_learning.py  # Transformers, Attention, ViT
│   │   └── time_series.py         # ARIMA/Prophet/LSTM
│   │
│   ├── 🔧 Infrastructure
│   │   ├── api.py                  # API REST
│   │   ├── streamlit_app.py       # Interface web
│   │   ├── config.py               # Configurações
│   │   └── schemas.py              # Schemas Pydantic
│   │
│   └── 🗂️ Management
│       ├── db_delete.py            # Exclusão de experimentos
│       └── registry_delete.py      # Exclusão de modelos registrados
│
├── tests/                          # Testes unitários, integração e aceitação
├── data/                           # Datasets importados
├── artifacts/                      # Artefatos de experimentos
├── .env                            # Configurações locais
├── .env.example                    # Exemplo de configurações
├── requirements.txt                # Dependências básicas + visualizações
├── requirements-base.txt           # Dependências mínimas essenciais
├── requirements-full.txt           # Todas as dependências (produção completa)
├── requirements-dev.txt            # Dependências de desenvolvimento
├── install_optional.py             # Instalador guiado de dependências opcionais
├── start_mlflow.py                 # Script para iniciar MLflow UI
├── docker-run.sh                   # Script Docker para automação
├── k8s-deploy.sh                   # Script Kubernetes para deploy
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose configuration
├── .dockerignore                   # Docker ignore file
├── k8s/                            # Kubernetes manifests
│   ├── namespace.yaml              # Namespace definition
│   ├── app-deployment.yaml         # Streamlit app deployment
│   ├── mlflow-deployment.yaml      # MLflow deployment
│   ├── api-deployment.yaml         # API deployment
│   ├── persistent-volumes.yaml     # PVCs for data persistence
│   ├── ingress.yaml                # Ingress configuration
│   └── configmap.yaml              # Configuration maps
└── README.md                       # Este arquivo
```

---

## 🎯 Fluxo de Trabalho Recomendado

### 1. **Data Preparation**
- Upload do dataset CSV
- Data validation com Pandera schemas
- Versionamento com DVC

### 2. **Model Development**
- AutoML clássico para baseline
- Hyperparameter optimization com Optuna
- Deep Learning (TensorFlow/PyTorch)
- Time Series forecasting

### 3. **Model Management**
- Experiment tracking completo
- Model Registry com versionamento
- Fine-tuning de hiperparâmetros

### 4. **Testing & Validation**
- Testes individuais e em lote
- Model validation automatizada
- Performance monitoring

### 5. **Production Deployment**
- API REST para predições
- Real-time monitoring
- Alert system
- Drift detection

---

## 🧪 Testes

### Executar todos os testes
```bash
pytest
```

### Testes específicos
```bash
pytest tests/unit/          # Testes unitários
pytest tests/integration/    # Testes de integração
pytest tests/acceptance/     # Testes de aceitação
```

---

## 🚀 Advanced Deep Learning

### **🤖 TabTransformer**
Modelo Transformer especializado para dados tabulares com:
- **Embeddings automáticos** para features categóricas
- **Multi-head attention** para capturar relações complexas
- **Layer normalization** para treinamento estável
- **Performance superior** em dados mistos (numéricos + categóricos)

**Casos de uso:**
- Dados com muitas features categóricas
- Features com alta cardinalidade
- Relações complexas entre variáveis
- Dados tabulares estruturados

### **👁️ Vision Transformer (ViT)**
Adaptação do Vision Transformer para dados tabulares:
- **Patch-based approach** - divide features em "patches"
- **Global self-attention** - captura padrões globais
- **Position embeddings** - mantém informação posicional
- **Hierarchical features** - aprende representações em múltiplos níveis

**Casos de uso:**
- Dados com padrões espaciais/temporais
- Features altamente correlacionadas
- Problemas não-lineares complexos
- Dados de séries temporais

### **📊 MLflow Integration**
Experiment tracking profissional com:
- **Automatic logging** de parâmetros e métricas
- **Model registry** integrado
- **Artifact management** para modelos e logs
- **Web UI** para visualização de experimentos

**Como usar:**
```bash
# Iniciar MLflow UI
python start_mlflow.py

# Acessar interface web
http://localhost:5000
```

### **🔍 Model Explainability**
Interpretação de modelos com SHAP e Captum:
- **SHAP values** para feature importance
- **DeepLift** para modelos PyTorch
- **Gradient attribution** para entender decisões
- **Visualizações interativas** de importância

**Benefícios:**
- Transparência nas decisões do modelo
- Identificação de features importantes
- Conformidade regulatória (GDPR, etc.)
- Debugging e melhoria de modelos

---

## 🐳 Docker & Kubernetes

### **Por que usar Docker/Kubernetes?**
- ✅ **Consistência**: Mesmo ambiente em qualquer máquina
- ✅ **Portabilidade**: Roda em qualquer lugar com Docker/K8s
- ✅ **Escalabilidade**: Fácil escalar horizontalmente
- ✅ **Isolamento**: Dependências isoladas e reproduzíveis
- ✅ **Deploy**: Deploy automatizado e versionado

### **🐳 Docker Features**
- **Multi-service**: Streamlit + MLflow + API
- **Volumes persistentes**: Dados preservados entre restarts
- **Health checks**: Monitoramento automático de saúde
- **Environment variables**: Configuração externa
- **Optimized images**: Python slim base + cache eficiente

### **☸️ Kubernetes Features**
- **Auto-scaling**: HPA e VPA support
- **Self-healing**: Restart automático de pods falhos
- **Rolling updates**: Deploy sem downtime
- **Load balancing**: Distribuição automática de tráfego
- **Persistent storage**: PVCs para dados duráveis
- **Ingress**: Single endpoint com TLS

### **🔧 Arquitetura de Containers**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     MLflow      │    │   FastAPI       │
│   (UI Web)      │    │   (Tracking)    │    │   (REST API)    │
│   Port: 8501    │    │   Port: 5000    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Shared Data   │
                    │   Volumes:       │
                    │   • /data        │
                    │   • /artifacts   │
                    │   • /models      │
                    │   • /mlruns      │
                    └─────────────────┘
```

### **🚀 Quick Start Docker**
```bash
# 1. Build e run tudo
./docker-run.sh run

# 2. Acessar serviços
open http://localhost:8501  # Streamlit
open http://localhost:5000  # MLflow
open http://localhost:8000/docs  # API

# 3. Ver status
docker-compose ps

# 4. Ver logs
./docker-run.sh logs app
```

### **☸️ Quick Start Kubernetes**
```bash
# 1. Deploy no cluster
./k8s-deploy.sh deploy

# 2. Ver status
./k8s-deploy.sh status

# 3. Acessar (port-forward)
kubectl port-forward svc/free-mlops-app-service 8501:8501 -n free-mlops

# 4. Escalar
./k8s-deploy.sh scale free-mlops-app 3
```

---

## 🏗️ Arquitetura

### **Local-First Design**
- ✅ Zero dependência de cloud
- ✅ Dados sempre no seu controle
- ✅ Processamento local
- ✅ Privacidade garantida

### **Modular & Extensible**
- ✅ Arquitetura em módulos independentes
- ✅ Fácil adição de novas funcionalidades
- ✅ Plugins para diferentes frameworks
- ✅ API limpa e documentada

### **Enterprise Features**
- ✅ Monitoring em tempo real
- ✅ Versionamento completo
- ✅ Validação automática
- ✅ Alertas e notificações

---

## 🎚️ Configuração

### Variáveis de Ambiente (.env)
```bash
# Diretórios
DATA_DIR=./data
ARTIFACTS_DIR=./artifacts

# API
API_HOST=127.0.0.1
API_PORT=8000

# Database
DB_PATH=./free_mlops.db

# Streamlit
STREAMLIT_HOST=localhost
STREAMLIT_PORT=8501
```

---

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/amazing-feature`)
3. Commit suas mudanças (`git commit -m 'Add amazing feature'`)
4. Push para a branch (`git push origin feature/amazing-feature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🎉 Conclusão

**Free MLOps Platform** é uma solução **enterprise-grade**, **open-source** e **self-hosted** que oferece recursos comparáveis às principais plataformas comerciais, mas com:

- 💰 **Custo ZERO**
- 🔒 **100% controle dos dados**
- 🏠 **Self-hosted**
- 🚀 **Production-ready**
- 🧩 **Modular e extensível**

**Perfeita para empresas que querem poder e flexibilidade sem os custos e dependências das plataformas cloud!**
