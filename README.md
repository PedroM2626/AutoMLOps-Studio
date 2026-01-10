# 🎯 MLOps Enterprise - Universal Framework

## 📋 Visão Geral

Framework MLOps completo e universal para treinamento, rastreamento e deploy de modelos de Machine Learning com integração total com DagsHub + MLflow.

## 🚀 Recursos Principais

### ✅ **Módulos Disponíveis:**

#### **1. 🤖 Machine Learning Clássico**
- **Algoritmos**: RandomForest, LogisticRegression, SVM, etc.
- **Suporte**: Classificação e Regressão
- **Auto-detecção**: Dados tabulares e NLP (TF-IDF)
- **Rastreamento**: Métricas completas no DagsHub

#### **2. 📈 Time Series (Prophet)**
- **Framework**: Facebook Prophet
- **Funcionalidade**: Previsão de séries temporais
- **Dados**: Sintéticos ou reais
- **Exportação**: Modelo registrado no MLflow

#### **3. 🧬 Clustering (K-Means)**
- **Algoritmo**: K-Means com otimização automática
- **Métricas**: Silhouette Score
- **Visualização**: Plot PCA automático
- **Flexibilidade**: Dados numéricos ou fallback sintético

#### **4. 🖼️ Computer Vision (YOLOv8)**
- **Modelos**: YOLOv8 (classify, detect, segment)
- **Fine-tuning**: Transfer learning com dados customizados
- **Exportação**: ONNX e outros formatos
- **Versões**: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

#### **5. 🔍 Monitoramento (Evidently)**
- **Drift Detection**: Data drift e Target drift
- **Relatórios**: HTML interativos
- **Integração**: Log automático no MLflow
- **Alertas**: Configuráveis

#### **6. 🚀 Model Serving (FastAPI)**
- **API REST**: Auto-gerada para qualquer modelo
- **Deploy**: Docker-ready
- **Carregamento**: Dinâmico do MLflow Registry
- **Documentação**: OpenAPI/Swagger automática

#### **7. 🖥️ Dashboard Interativo (Streamlit)**
- **Análise de Dados**: Upload de CSV e análise exploratória automática.
- **Visualização**: Gráficos interativos com Plotly.
- **Gestão de Experimentos**: Visualização detalhada de resultados do MLflow.
- **Configuração**: Interface amigável para parâmetros do sistema.

---

## 📁 Estrutura do Projeto

```
free-mlops/
├── experiments/                    # Core do framework de treinamento
│   ├── train_and_save_professional.py
│   ├── main.py
│   ├── .env
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── app_serving.py
│   └── requirements.txt
└── README.md                       # Este arquivo
```

---

## 🛠️ Instalação e Configuração

### **Pré-requisitos:**
```bash
# Python 3.8+
pip install python>=3.8

# Git LFS (para modelos grandes)
git lfs install
```

### **Dependências Principais:**
```bash
# MLOps & Tracking
pip install mlflow dagshub optuna

# Machine Learning & Dashboard
pip install scikit-learn pandas numpy matplotlib streamlit plotly

# Deep Learning
pip install torch transformers datasets

# Time Series
pip install prophet

# Computer Vision
pip install ultralytics

# Monitoramento
pip install evidently

# Serving
pip install fastapi uvicorn python-dotenv
```

### **Configuração do Ambiente:**
```bash
# Copiar arquivo de ambiente
cp experiments/.env.example experiments/.env

# Editar configurações
nano experiments/.env
```

**Variáveis de ambiente (.env):**
```bash
DAGSHUB_REPO_OWNER=PedroM2626
DAGSHUB_REPO_NAME=free-mlops
DAGSHUB_TOKEN=seu_token_aqui
MLFLOW_TRACKING_URI=https://dagshub.com/PedroM2626/free-mlops.mlflow
```

---

## 🚀 Uso Rápido

### **1. Executar o Dashboard (Streamlit):**
```bash
streamlit run streamlit_app/app_refactored.py
```

### **2. Executar Todos os Módulos de Treinamento:**
```bash
cd experiments
python train_and_save_professional.py --task all
```

### **3. Executar Módulo Específico:**
```bash
# Machine Learning Clássico
python train_and_save_professional.py --task classic

# Time Series
python train_and_save_professional.py --task ts

# Clustering
python train_and_save_professional.py --task cluster

# Computer Vision
python train_and_save_professional.py --task cv
```

---

## 📊 Resultados no DagsHub

### **🔗 Experimentos Criados:**
- **`/classic_classification`**: Modelos de classificação clássicos
- **`/classic_regression`**: Modelos de regressão clássicos
- **`/time_series`**: Modelos Prophet
- **`/clustering`**: Modelos K-Means
- **`/cv_detect`**: YOLO detecção
- **`/cv_classify`**: YOLO classificação
- **`/cv_segment`**: YOLO segmentação

### **📁 Artefatos Salvos:**
- **Modelos**: `.pkl`, `.pt`, `.onnx`
- **Métricas**: JSON com todas as métricas
- **Visualizações**: PNG (matriz confusão, PCA plots)
- **Configurações**: YAML com hiperparâmetros
- **Ambiente**: `requirements.txt`, `conda.yaml`

---

## 🐳 Docker e Deploy

### **Build da Imagem:**
```bash
docker build -t mlops-enterprise .
```

### **Executar com Docker Compose:**
```bash
docker-compose up -d
```

### **Deploy da API (FastAPI):**
```bash
# Gerar API automaticamente
python experiments/train_and_save_professional.py --task all
# Isso cria app_serving.py

# Iniciar servidor
uvicorn experiments.app_serving:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Testes

### **Executar Testes:**
```bash
cd experiments
python -m pytest tests/ -v
```

---

## 🤝 Contribuição

### **📋 Como Contribuir:**
1. Fork do projeto
2. Criar feature branch
3. Implementar mudanças
4. Adicionar testes
5. Submeter Pull Request

---

## 📝 Licença

MIT License - Ver arquivo LICENSE para detalhes.

---

## 🎯 Roadmap

### **✅ Implementado:**
- [x] ML Clássico com DagsHub
- [x] Time Series (Prophet)
- [x] Clustering (K-Means)
- [x] Computer Vision (YOLOv8)
- [x] Monitoramento (Evidently)
- [x] Model Serving (FastAPI)
- [x] Dashboard Interativo (Streamlit)
- [x] Dockerização

---

## 🆘 Suporte

### **📋 Problemas Comuns:**
1. **DagsHub Connection**: Verificar token e permissões
2. **CUDA Memory**: Reduzir batch size ou usar CPU
3. **Dependencies**: Usar requirements.txt exato
4. **Port Conflicts**: Mudar portas no docker-compose.yml

**🎉 Framework MLOps Enterprise completo e pronto para uso!**
