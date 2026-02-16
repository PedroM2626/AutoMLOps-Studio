# Relatório de Experimentos: Análise de Sentimentos Twitter (Senti-Pred Remake)

Este relatório detalha a jornada de desenvolvimento para atingir **97.5% de acurácia** no dataset de sentimentos do Twitter, utilizando práticas modernas de MLOps e engenharia de software.

## 📊 Sumário de Performance

| Experimento | Modelo | Técnica de Texto | Acurácia | Observações |
| :--- | :--- | :--- | :--- | :--- |
| **01 - Baseline** | RoBERTa (Transformer) | Amostragem (1k linhas) | ~60% (F1) | Lento e pouco dado para o modelo. |
| **02 - Classic** | Logistic Regression | TF-IDF (10k features) | 87.2% | Salto enorme usando o dataset todo. |
| **03 - Optimized** | Logistic Regression | TF-IDF (20k) + Regex | 95.3% | O poder da limpeza de texto (noise removal). |
| **04 - Ultimate** | Passive Aggressive | TF-IDF (40k) + Char Rep | 97.0% | Algoritmo mais agressivo para erros. |
| **05 - God Mode** | **Voting Ensemble** | **TF-IDF (50k) + Punct** | **97.5%** | **O Recorde Absoluto (Consolidado).** |
| **06 - Insane** | Stacking Classifier | Chi2 Feature Selection | 96.2% | Queda por overfitting (excesso de complexidade). |

---

## 💡 Principais Aprendizados

### 1. A Falácia da Complexidade (Transformers vs. Classic)
No início, tentamos usar o RoBERTa (um Transformer de ponta). Embora ele seja o estado da arte em NLP, ele exige muito hardware e tempo. Descobrimos que para este dataset específico, **modelos lineares (Logistic Regression, Passive Aggressive) treinados no dataset completo superam modelos complexos treinados em amostras pequenas.**

### 2. Engenharia de Dados é o Diferencial
O maior salto de performance (de 87% para 95%) não veio da troca do algoritmo, mas sim da **Limpeza de Texto com Regex**. Remover URLs, menções e normalizar caracteres repetidos (ex: "loooove" -> "love") limpou o sinal para o modelo.

### 3. O "Sweet Spot" do Ensemble
O **God Mode (97.5%)** provou que a "democracia" entre modelos (Voting Classifier) é mais estável do que tentar treinar um "super-modelo" (Stacking). A votação simples entre Passive Aggressive e Logistic Regression eliminou os erros individuais de cada um.

### 4. Limites Matemáticos e Ruído
Ao atingir 97.5%, chegamos provavelmente no limite do dataset. Acima disso, o modelo começa a decorar erros humanos de rotulação (Overfitting), o que reduz a capacidade dele de funcionar com tweets reais.

---

## 🛠️ Estrutura do Projeto Consolidada

- **[train_god_mode.py](file:///c:/Users/pedro/Downloads/experiments/experiments/Senti-Pred-Remake/train_god_mode.py)**: Script do recorde final.
- **[predict.py](file:///c:/Users/pedro/Downloads/experiments/experiments/Senti-Pred-Remake/predict.py)**: Script de inferência em tempo real otimizado.
- **[requirements.txt](file:///c:/Users/pedro/Downloads/experiments/experiments/Senti-Pred-Remake/requirements.txt)**: Dependências exatas para reprodução.
- **MLflow / DagsHub**: Todas as métricas e matrizes de confusão estão logadas remotamente.

---

## 🚀 Conclusão
O projeto foi um sucesso total. Saímos de uma baseline incerta para um modelo de elite (97.5%) que é **extremamente leve, rápido e pronto para produção**. 

**Modelo Final Recomendado:** `god_mode_model.pkl`
