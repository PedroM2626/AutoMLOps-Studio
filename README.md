# Free MLOps (MVP)

Plataforma **local-first**, **100% gratuita** e **self-hosted** (sem cloud obrigatória) para criar, treinar, avaliar e fazer deploy local de modelos de Machine Learning **sem escrever código**, via interface visual.

Este repositório implementa o **MVP tabular**:

- Upload de dataset **CSV**
- Escolha de alvo (target) e tipo do problema (classificação/regressão)
- Treinamento automático (AutoML lite com scikit-learn)
- Comparação de modelos (leaderboard) + métricas
- Tracking/versionamento de experimentos (SQLite)
- Export do modelo treinado (`.pkl`)
- Deploy local simples via API REST (FastAPI)

## Requisitos

- Python **3.10+**

## Instalação

1) (Opcional, recomendado) crie e ative um ambiente virtual.

2) Instale as dependências:

```bash
pip install -r requirements.txt
```

Para rodar testes:

```bash
pip install -r requirements-dev.txt
```

## Configuração (.env)

Este projeto lê configurações de um arquivo `.env` na raiz.

- Arquivo usado no desenvolvimento: `.env`
- Exemplo: `.env.example`

Por padrão, tudo roda localmente usando:

- `./data` (datasets importados)
- `./artifacts` (artefatos/experimentos)
- `./free_mlops.db` (SQLite)

## Como rodar a interface (no-code)

Na raiz do repositório:

```bash
streamlit run free_mlops/streamlit_app.py
```

Fluxo recomendado:

- Upload do CSV
- Seleção do target
- Seleção do tipo (classificação/regressão)
- Treinar
- Ver leaderboard e métricas
- Baixar o `model.pkl`

## Como rodar a API (deploy local)

Na raiz do repositório:

```bash
python -m free_mlops.api
```

A API sobe por padrão em:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

### Endpoints principais

- `GET /health`
- `GET /models` (lista de experimentos/modelos disponíveis)
- `POST /predict` (predição usando o modelo mais recente, ou um `experiment_id` específico)

Exemplo de request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"col_a": 1, "col_b": "x"}]}'
```

## Estrutura do projeto

```text
free-mlops/
  free_mlops/
    api.py
    automl.py
    config.py
    db.py
    schemas.py
    service.py
    streamlit_app.py
  tests/
    unit/
    integration/
    acceptance/
  .env
  .env.example
  .gitignore
  requirements.txt
  requirements-dev.txt
  README.md
```

## Testes

Na raiz do repositório:

```bash
pytest
```

## Notas do MVP

- Foco em dados **tabulares** (CSV).
- Não depende de GPU.
- Não requer cloud.
- O AutoML é propositalmente simples e transparente (candidatos scikit-learn + avaliação em holdout).
