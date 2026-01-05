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
- **Time Series:** ARIMA, Prophet, LSTM para séries temporais
- **Multi-framework:** suporte a múltiplos frameworks de ML/DL
- **Auto-tuning:** configuração automática de hiperparâmetros

---

## 🏆 Comparação com Plataformas Comerciais

| Funcionalidade | Free MLOps | Azure ML | SageMaker | Vertex AI | Databricks |
|---|---|---|---|---|---|
| **AutoML Clássico** | ✅ Scikit-learn | ✅ | ✅ | ✅ | ✅ |
| **Deep Learning** | ✅ TF/PyTorch | ✅ | ✅ | ✅ | ✅ |
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

### Dependências Opcionais (por funcionalidade)

#### 🤖 Deep Learning
```bash
pip install tensorflow  # Para modelos TensorFlow
pip install torch        # Para modelos PyTorch
```

#### 📈 Time Series
```bash
pip install statsmodels pmdarima  # Para ARIMA
pip install prophet                # Para Prophet
pip install tensorflow              # Para LSTM (já incluído acima)
```

#### 🔬 Hyperparameter Optimization
```bash
pip install optuna  # Para otimização avançada
```

#### 📦 Data Versioning & Validation
```bash
pip install dvc      # Para versionamento de dados
pip install pandera  # Para validação de dados
```

#### 📊 Visualizações Avançadas
```bash
pip install plotly  # Para gráficos interativos (já incluído)
```

### Instalação Completa (todas as funcionalidades)
```bash
pip install -r requirements.txt
pip install tensorflow torch optuna statsmodels pmdarima prophet dvc pandera plotly
```

---

## 🎮 Como Usar

### Interface Web (Recomendado)
```bash
streamlit run free_mlops/streamlit_app.py
```

Acesse: `http://localhost:8501`

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
│   │   ├── deep_learning.py       # TensorFlow/PyTorch
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
├── requirements.txt                # Dependências base
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
