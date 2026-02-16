# Senti-Pred: Scripts Python Modulares

Este guia detalha como executar o projeto Senti-Pred utilizando os scripts Python modulares localizados no diretório `src/scripts/`.

## Visão Geral

Esta abordagem é ideal para quem busca uma execução mais controlada, modular e que pode ser facilmente integrada em outros sistemas ou fluxos de trabalho automatizados. Os scripts cobrem desde a análise exploratória de dados até o treinamento do modelo e avaliação final.

## Estrutura dos Scripts

-   `src/scripts/01_eda.py`: Análise Exploratória de Dados (EDA) - gera gráficos de distribuição e visualizações iniciais
-   `src/scripts/02_preprocessing.py`: Realiza o pré-processamento e limpeza dos dados em inglês
-   `src/scripts/03_modeling.py`: Treina múltiplos modelos de classificação e salva o melhor
-   `src/scripts/04_evaluation.py`: Avalia o desempenho do modelo treinado com métricas detalhadas

## Fluxo de Execução

Os scripts devem ser executados em ordem sequencial, pois cada um depende dos artefatos gerados pelo anterior:

### 1. Análise Exploratória de Dados (EDA)

```bash
python src/scripts/01_eda.py
```

**O que faz:**
- Carrega os dados brutos (`twitter_training.csv` e `twitter_validation.csv`)
- Gera análises estatísticas e visualizações
- Salva gráficos em `reports/visualizacoes/`

**Artefatos gerados:**
- Gráficos de distribuição de sentimentos por split
- Análise de comprimento de textos
- Top palavras mais frequentes

### 2. Pré-processamento de Dados

```bash
python src/scripts/02_preprocessing.py
```

**O que faz:**
- Limpa e normaliza os textos (remove URLs, menções, hashtags, números)
- Remove stopwords em inglês
- Aplica lematização com POS tagging
- Salva dados processados em formato pickle

**Artefatos gerados:**
- `data/processed/processed_data.pkl` - contém DataFrames processados para treino e validação

### 3. Treinamento do Modelo

```bash
python src/scripts/03_modeling.py
```

**O que faz:**
- Carrega dados processados do pickle
- Treina três modelos diferentes: LogisticRegression, MultinomialNB e LinearSVC
- Compara performance usando F1-score macro
- Salva o melhor modelo e métricas detalhadas

**Modelos testados:**
- **LogisticRegression**: Regressão logística multiclasse
- **MultinomialNB**: Naive Bayes para features de texto
- **LinearSVC**: SVM linear otimizado para grandes conjuntos de dados

**Artefatos gerados:**
- `src/models/sentiment_model.pkl` - melhor modelo salvo (joblib)
- `reports/metrics/model_metrics.json` - métricas detalhadas de todos os modelos
- Gráficos comparativos (ROC, Precision-Recall, Matrizes de Confusão) em `reports/visualizacoes/`

### 4. Avaliação Final

```bash
python src/scripts/04_evaluation.py
```

**O que faz:**
- Carrega o modelo treinado e dados processados
- Gera métricas finais de avaliação
- Cria visualizações adicionais de desempenho

**Artefatos gerados:**
- Métricas finais consolidadas
- Matriz de confusão detalhada
- Análises adicionais de performance

## Requisitos e Dependências

Certifique-se de ter instalado todas as dependências do projeto:

```bash
pip install -r requirements.txt
```

Ou instale as dependências específicas dos scripts:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk joblib
```

## Download de Recursos NLTK

Os scripts fazem download automático dos recursos necessários do NLTK, mas se encontrar problemas, execute manualmente:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```

## Estrutura Esperada de Dados

Os scripts esperam encontrar os dados brutos no seguinte formato:

**Arquivos:**
- `data/raw/twitter_training.csv`
- `data/raw/twitter_validation.csv`

**Formato CSV:** (sem header, 4 colunas)
```
tweet_id,entity,sentiment,text
```

**Classes de sentimento:**
- `Positive`: Sentimento positivo
- `Negative`: Sentimento negativo  
- `Neutral`: Sentimento neutro
- `Irrelevant`: Fora do escopo (opcional)

## Saída e Resultados

Após executar todos os scripts, você terá:

### Modelo Treinado
- **Localização:** `src/models/sentiment_model.pkl`
- **Tipo:** Pipeline scikit-learn completo (TF-IDF + Classificador)
- **Melhor modelo:** LinearSVC (baseado em F1-score macro)

### Métricas de Performance
- **Localização:** `reports/metrics/model_metrics.json`
- **Inclui:** Acurácia, F1-score, ROC-AUC, Precision-Recall para cada modelo
- **Tempo de treinamento:** Registrado para análise de eficiência

### Visualizações
- **Localização:** `reports/visualizacoes/`
- **Gráficos:** Distribuições, matrizes de confusão, curvas ROC/PR, análises de textos

## Solução de Problemas

### Erro: "Processed data not found"
Execute o script `02_preprocessing.py` antes do `03_modeling.py`

### Erro: "Model not found"  
Execute o script `03_modeling.py` antes do `04_evaluation.py`

### Erro: Recursos NLTK não encontrados
Execute o download manual dos pacotes NLTK conforme mostrado acima

### Performance lenta
- Considere reduzir `max_features` no TfidfVectorizer
- Use uma amostra menor dos dados para testes
- Verifique se tem memória RAM suficiente (recomendado: 8GB+)

## Integração com Outros Componentes

Os modelos e métricas gerados podem ser usados por:
- **Dashboard Streamlit:** `streamlit_dashboard/app.py` (agora com predição interativa e em lote)
- **API Django:** `src/api/views.py`
- **Dashboard R Shiny:** `r_shiny/app.py`

## Contribuição

Consulte o [README.md](../../README.md) principal para diretrizes de contribuição.

## Licença

Consulte o [README.md](../../README.md) principal para informações sobre a licença.