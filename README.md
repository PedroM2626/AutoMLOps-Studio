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

Este documento serve como referência central para todas as funcionalidades, opções de configuração e aprendizados técnicos desenvolvidos durante a criação do projeto.

---

## 🎯 Objetivo e Problemática

Muitas vezes, aprender Machine Learning parece fragmentado entre teoria e código complexo. Este projeto resolve isso ao centralizar:
- **Aprendizado Prático**: Entenda como o pré-processamento, o treinamento e o monitoramento se conectam.
- **Prototipagem Rápida**: Teste ideias de modelos em segundos com arquivos CSV ou imagens.
- **Desmistificação de MLOps**: Veja na prática como o versionamento de modelos (MLflow), integração com DagsHub e a detecção de desvios (Drift) funcionam em um fluxo real.

## 👥 Público Alvo

- **Estudantes de Ciência de Dados**: Que querem ver a teoria aplicada em uma interface visual.
- **Curiosos e Entusiastas de ML**: Que buscam uma ferramenta ágil para explorar datasets sem barreiras técnicas.
- **Desenvolvedores em Aprendizado**: Que desejam entender como integrar modelos de ML em APIs e Dashboards de forma simplificada.

---

## ✨ Funcionalidades e Detalhes Técnicos

### 1. Gestão de Dados (Data Lake)
- **Upload de Dados:** Suporte para arquivos CSV.
- **Data Lake Local:** Armazenamento versionado de datasets (raw/processed).
- **Carregamento de Dados:** Seleção de datasets e versões específicas para o workspace de trabalho.

### 2. Configuração de Treino (AutoML)

#### 2.1. Definição da Tarefa
O sistema suporta os seguintes tipos de problemas de Machine Learning:
- **Classification:** Previsão de classes discretas (ex: fraude/não fraude).
- **Regression:** Previsão de valores contínuos (ex: preço de imóveis).
- **Clustering:** Agrupamento não supervisionado.
- **Time Series:** Previsão temporal (ex: vendas futuras).
- **Anomaly Detection:** Detecção de outliers.

#### 2.2. Fonte do Modelo
- **AutoML Standard:** Utiliza bibliotecas padrão (Scikit-Learn, XGBoost, Transformers).
- **Model Registry:** Permite selecionar um modelo previamente treinado e registrado para *fine-tuning* ou re-treino.
- **Upload Local (.pkl):** Permite carregar um modelo serializado externamente.

#### 2.3. Seleção de Modelos
- **Automático (Preset):** O sistema escolhe os melhores candidatos.
- **Manual (Selecionar):** O usuário escolhe especificamente quais algoritmos testar (ex: Random Forest, XGBoost, SVM).
- **Custom Ensemble Builder:**
    - **Voting:** Combina predições por voto majoritário (Hard) ou média de probabilidades (Soft). Suporta pesos customizados.
    - **Stacking:** Treina um "Meta-Modelo" (ex: Regressão Logística) que aprende a combinar as saídas dos modelos base.

#### 2.4. Otimização de Hiperparâmetros (HPO)
O sistema utiliza **Optuna** como motor de otimização, oferecendo quatro modos selecionáveis manualmente:
- **Bayesian Optimization (TPE):** (Padrão) Utiliza o estimador *Tree-structured Parzen Estimator* para focar nas áreas promissoras do espaço de busca. Mais eficiente que Random/Grid.
- **Random Search:** Exploração aleatória do espaço de busca, ideal para benchmarks.
- **Grid Search:** Busca exaustiva em uma grade pré-definida. (Implementado via amostragem controlada no Optuna para garantir cobertura).
- **Hyperband:** Técnica avançada que descarta configurações ruins rapidamente (early stopping agressivo), permitindo testar muito mais combinações em menos tempo.

#### 2.5. Presets de Treino (AutoML)
Para agilidade, o sistema oferece perfis pré-configurados (`automl_engine.py`):
- **Fast:** ~15 trials. Foca em modelos leves (Logistic Regression, Random Forest) com validação simples. Ideal para testes rápidos.
- **Medium:** ~40 trials. Inclui modelos de Gradient Boosting (XGBoost, LightGBM) e validação cruzada mais robusta (CV=5).

#### 2.6. Validação Automática Inteligente
Define como os modelos são avaliados para evitar *overfitting*. O sistema conta com um modo **Automático Inteligente**:
- **Automático (Recomendado):**
    - **Séries Temporais:** Detecta automaticamente e aplica `TimeSeriesSplit`.
    - **Pequenos Datasets (<1000 amostras):** Aplica `Cross-Validation` para garantir robustez.
    - **Grandes Datasets (>=1000 amostras):** Aplica `Holdout (Train-Test Split)` para eficiência.
- **Modos Manuais:**
    - **K-Fold Cross Validation**
    - **Stratified K-Fold** (Apenas Classificação)
    - **Holdout**
    - **Time Series Split**

### 3. 👁️ Visão Computacional (CV Engine)
O módulo `cv_engine.py` expande as capacidades para Deep Learning e Visão Computacional:
- **Tarefas Suportadas:**
    - **Classificação de Imagens:** Identificação de classes (ex: Gato vs Cachorro).
    - **Segmentação Semântica:** Classificação pixel a pixel (ex: separar fundo e objeto) usando DeepLabV3.
    - **Detecção de Objetos:** Localização com Bounding Boxes usando Faster R-CNN.
- **Modelos & Transfer Learning:**
    - **ResNet18 / ResNet50:** Arquiteturas robustas para classificação geral.
    - **MobileNetV2:** Otimizado para eficiência e dispositivos móveis.
    - **Backbones:** Pesos pré-treinados no ImageNet para convergência rápida.

### 4. ⚖️ Análise de Estabilidade e Robustez
A aba de **Estabilidade** permite avaliar a confiabilidade dos modelos gerados através de testes rigorosos:
- **Robustez a Variação de Dados**: Testa o modelo em múltiplos splits de treino/teste para verificar a consistência das métricas.
- **Robustez à Inicialização**: Avalia o impacto de diferentes sementes aleatórias (seeds) no treinamento.
- **Sensibilidade a Hiperparâmetros**: Analisa como a performance varia ao alterar um hiperparâmetro específico.
- **Análise Geral**: Executa uma bateria completa de testes e gera um relatório unificado de estabilidade.

### 5. MLOps, API e Integrações
- **MLflow Integration:** Rastreamento completo de experimentos (parâmetros, métricas, artefatos).
- **DagsHub Connection:**
    - Sincronização com repositórios remotos DagsHub.
    - Autenticação via Token.
    - Visualização de status de conexão em tempo real.
- **Drift Detection:** Monitoramento de desvio de dados entre treino e produção (Data Drift).
- **Model Registry:** Versionamento e gestão de estágios de modelos (Staging, Production, Archived).
- **Explicabilidade**: Integração nativa com SHAP.
- **Docker Ready**: Ambiente containerizado pronto para uso.
- **API Serving (FastAPI):**
    - Módulo `api.py` fornece uma interface REST robusta.
    - **Endpoints:** `/predict` para inferência e `/` para health check.
    - **Segurança:** Autenticação via `x-api-key` no header.
    - **Auto-Reload:** Carrega automaticamente o modelo mais recente salvo em `models/`.

---

## 🧠 Aprendizados e Decisões Técnicas

### 1. Flexibilidade com Optuna
Optamos pelo **Optuna** em vez do `GridSearchCV` do Scikit-Learn devido à sua arquitetura "define-by-run". Isso permitiu:
- Implementar *Bayesian Optimization* facilmente.
- Simular *Grid Search* e *Random Search* apenas alterando o `sampler` (TPESampler, RandomSampler, GridSampler).
- Integrar *Pruning* (Hyperband) para interromper treinos ruins cedo, economizando recursos computacionais.

### 2. Desafios do Grid Search em Espaços Contínuos
Aprendemos que o *Grid Search* tradicional é incompatível com distribuições contínuas (ex: `loguniform` para learning rate).
- **Solução:** Quando o usuário seleciona "Grid Search", o sistema restringe o espaço de busca a um conjunto finito de valores discretos ou reverte para *Random Search* com alta contagem de tentativas se o espaço for muito complexo.

### 3. Validação Automática Inteligente
Implementamos uma lógica de decisão para a validação automática (`validation_strategy='auto'`):
- **Time Series:** Sempre usa `TimeSeriesSplit`.
- **Dados Pequenos (< 1000 amostras):** Usa `Cross-Validation` (CV) para maior robustez estatística.
- **Dados Grandes (>= 1000 amostras):** Usa `Holdout` para eficiência computacional, já que a variância da estimativa de erro diminui com o volume de dados.

### 4. Persistência e Estado na Interface (Streamlit)
O Streamlit reexecuta o script a cada interação. Para manter conexões (como DagsHub) e configurações:
- Usamos `st.session_state` para variáveis temporárias.
- Usamos `os.environ` para credenciais e URIs do MLflow, garantindo que o `automl_engine.py` (que roda em outro processo ou contexto) tenha acesso às configurações definidas na UI.

### 5. Integração Híbrida MLflow (Local vs Remoto)
- **SQLite (Local):** Ótimo para desenvolvimento rápido e sem internet, mas tem problemas de *locking* com múltiplas threads.
- **DagsHub (Remoto):** Resolve a colaboração e visualização, mas requer tratamento de erros de rede e autenticação.
- **Solução:** Criamos um "switch" na interface que altera dinamicamente a `MLFLOW_TRACKING_URI` e recarrega o cliente MLflow sem precisar reiniciar a aplicação.

### 6. Separação de Responsabilidades
- `app.py`: Apenas UI e captura de input.
- `automl_engine.py`: Lógica pesada de ML, independente da UI.
- `mlops_utils.py`: Funções utilitárias reutilizáveis.
Isso facilitou a criação de scripts de teste (`test_interface_simulation.py`) que validam o motor de ML sem precisar clicar na interface.

---

## 📂 Estrutura do Projeto

- `app.py`: Interface principal (Streamlit Dashboard).
- `automl_engine.py`: Motor de AutoML (Treinamento, Otimização, Validação).
- `cv_engine.py`: Motor de Visão Computacional.
- `mlops_utils.py`: Utilitários de MLOps.
- `api.py`: API de serving (FastAPI).
- `test_interface_simulation.py`: Script de teste para validação das funcionalidades de otimização e interface.
- `docker-compose.yml` & `Dockerfile`: Configuração de containers.

---

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

### 🖥️ Aplicação Desktop (Electron)

Você pode executar o projeto como uma aplicação desktop híbrida (Electron + Python).

1. **Pré-requisitos**: Certifique-se de ter o `Node.js` e `npm` instalados.
2. **Instale as dependências do Electron**:
   ```bash
   npm install
   ```
3. **Inicie em modo de desenvolvimento**:
   ```bash
   npm start
   ```
   Isso iniciará o servidor Python em segundo plano e abrirá a janela do Electron.

4. **Build do Executável**:
   Para criar um instalador (.exe, .dmg, .AppImage):
   ```bash
   npm run dist
   ```

---

## 🧪 Testes e Validação

Para verificar se todas as funcionalidades de otimização (Grid, Random, Bayesian, Hyperband) e validação automática estão funcionando corretamente, execute o script de simulação:

```bash
python test_interface_simulation.py
```
Este script simula o comportamento da interface utilizando os datasets disponíveis no `data_lake`.

---

## 🔮 Próximos Passos Sugeridos
*   **Deploy Automatizado:** Gerar containers Docker com o modelo treinado (servindo via API REST/FastAPI) com um clique.
*   **Explainability (XAI):** Adicionar SHAP/LIME na aba de experimentos para explicar as decisões dos modelos.
*   **Pipeline de Retreino:** Configurar Jobs agendados para verificar Drift e disparar retreino automático.

---

**Desenvolvido por Pedro Morato Lahoz.**
