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

---

## 📁 Estrutura do Projeto

```
free-mlops/
├── experiments/
│   ├── train_and_save_professional.py    # Framework principal
│   ├── main.py                        # Entry point FLAML/AutoGluon
│   ├── .env                          # Configurações de ambiente
│   ├── Dockerfile                     # Containerização
│   ├── docker-compose.yml              # Orquestração
│   ├── app_serving.py               # API de serving gerada
│   ├── requirements.txt               # Dependências
│   ├── src/                         # Módulos auxiliares
│   │   ├── utils.py
│   │   ├── flaml_train.py
│   │   └── autogluon_train.py
│   └── tests/                       # Testes automatizados
└── README.md                        # Este arquivo
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

# Machine Learning
pip install scikit-learn pandas numpy matplotlib

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
cp .env.example .env

# Editar configurações
nano .env
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

### **1. Executar Todos os Módulos:**
```bash
cd experiments
python train_and_save_professional.py --task all
```

### **2. Executar Módulo Específico:**
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

### **3. Exemplos de Uso:**

#### **🤖 Machine Learning Clássico:**
```python
from experiments.train_and_save_professional import MLOpsEnterprise

# Inicializar framework
ml = MLOpsEnterprise()

# Treinar modelo de classificação
ml.train_classic_ml(task='classification', data_path='seus_dados.csv')

# Treinar modelo de regressão
ml.train_classic_ml(task='regression', data_path='seus_dados.csv')
```

#### **🧬 Clustering:**
```python
# Treinar K-Means com 5 clusters
ml.train_clustering(n_clusters=5, data_path='seus_dados.csv')

# Resultados salvos automaticamente no DagsHub
# - Modelo K-Means
# - Plot PCA visualização
# - Silhouette Score
```

#### **🖼️ Computer Vision:**
```python
# Treinar YOLO para detecção
ml.train_cv(
    task='detect',
    data_config='path/to/dataset.yaml',
    model_type='yolov8n.pt',
    epochs=50
)

# Treinar YOLO para classificação
ml.train_cv(
    task='classify',
    data_config='path/to/dataset.yaml',
    model_type='yolov8s.pt',
    epochs=30
)
```

#### **📈 Time Series:**
```python
# Com dados reais
ml.train_time_series(data_path='vendas_mensais.csv')

# Com dados sintéticos (para testes)
ml.train_time_series()
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

### **🎯 Model Registry:**
- **`classic_classification_model`**: Melhor modelo de classificação
- **`classic_regression_model`**: Melhor modelo de regressão
- **`ts_prophet_model`**: Modelo Prophet
- **`clustering_model`**: Modelo K-Means
- **`cv_yolo_model`**: Modelo YOLO

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

### **Deploy da API:**
```bash
# Gerar API automaticamente
python train_and_save_professional.py --task all
# Isso cria app_serving.py

# Iniciar servidor
uvicorn app_serving:app --host 0.0.0.0 --port 8000

# Ou com Docker
docker run -p 8000:8000 mlops-enterprise
```

---

## 📈 Monitoramento e Otimização

### **🔍 Detecção de Drift:**
```python
# Comparar dados de referência vs atuais
ml.detect_drift(
    reference_df=dados_treino,
    current_df=dados_producao
)

# Relatório gerado automaticamente no DagsHub
```

### **⚡ Otimização com Optuna:**
```python
# Framework já integrado com Optuna
# Hiperparâmetros otimizados automaticamente
# Resultados logados no MLflow
```

---

## 🧪 Testes

### **Executar Testes:**
```bash
cd experiments
python -m pytest tests/ -v
```

### **Testes de Integração:**
```bash
# Testar conexão DagsHub
python -c "from experiments.train_and_save_professional import MLOpsEnterprise; MLOpsEnterprise()"

# Testar todos os módulos
python train_and_save_professional.py --task all
```

---

## 🔧 Configurações Avançadas

### **Customizar Modelos:**
```python
# Configuração customizada para clustering
ml.train_clustering(
    n_clusters=10,
    data_path='custom_data.csv'
)

# Configuração customizada para CV
ml.train_cv(
    task='detect',
    data_config='custom_dataset.yaml',
    model_type='yolov8l.pt',
    epochs=100
)
```

### **Integração CI/CD:**
```yaml
# .github/workflows/mlflow.yml
name: MLOps Pipeline
on: [push]
jobs:
  mlflow:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run MLOps Pipeline
        run: python experiments/train_and_save_professional.py --task all
        env:
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
```

---

## 📚 Documentação e Recursos

### **🔗 Links Úteis:**
- **DagsHub**: https://dagshub.com/PedroM2626/free-mlops
- **MLflow**: https://dagshub.com/PedroM2626/free-mlops.mlflow
- **Documentação**: https://docs.dagshub.com
- **Prophet**: https://facebook.github.io/prophet/
- **YOLOv8**: https://docs.ultralytics.com/
- **Evidently**: https://evidentlyai.com/

### **📖 Tutoriais:**
1. **Setup Inicial**: Configuração do ambiente
2. **Primeiro Experimento**: ML clássico
3. **Computer Vision**: Treinar YOLO
4. **Time Series**: Previsão com Prophet
5. **Clustering**: K-Means avançado
6. **Deploy**: API em produção
7. **Monitoramento**: Detecção de drift

---

## 🤝 Contribuição

### **📋 Como Contribuir:**
1. Fork do projeto
2. Criar feature branch
3. Implementar mudanças
4. Adicionar testes
5. Submeter Pull Request

### **🏗️ Arquitetura:**
- **Modular**: Cada módulo independente
- **Extensível**: Fácil adicionar novos algoritmos
- **Testável**: Cobertura completa de testes
- **Documentado**: Código auto-explicativo

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
- [x] Dockerização

### **🚧 Próximo:**
- [ ] Integração com mais frameworks (HuggingFace, Weights & Biases)
- [ ] AutoML avançado (Auto-sklearn, TPOT)
- [ ] Model explainability (SHAP, LIME)
- [ ] Distributed training
- [ ] Kubernetes deployment
- [ ] Real-time monitoring dashboard

---

## 🆘 Suporte

### **📋 Problemas Comuns:**
1. **DagsHub Connection**: Verificar token e permissões
2. **CUDA Memory**: Reduzir batch size ou usar CPU
3. **Dependencies**: Usar requirements.txt exato
4. **Port Conflicts**: Mudar portas no docker-compose.yml

### **📞 Contato:**
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [seu-email]

---

**🎉 Framework MLOps Enterprise completo e pronto para uso!**

**Todos os módulos integrados com DagsHub + MLflow para rastreamento completo e versionamento automático.**
