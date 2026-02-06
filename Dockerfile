# Usar uma imagem base leve com Python 3.11
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para algumas bibliotecas (ex: LightGBM, XGBoost, OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar os arquivos de dependências
COPY requirements.txt .

# Instalar as dependências do Python
# Usando --no-cache-dir para reduzir o tamanho da imagem
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código do projeto
COPY . .

# Criar diretórios para persistência
RUN mkdir -p mlruns data_lake models

# Expor as portas necessárias
# 8000: FastAPI, 8501: Streamlit
EXPOSE 8000 8501

# O comando padrão será sobrescrito pelo Docker Compose
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
