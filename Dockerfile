# Usar uma imagem base leve com Python 3.11
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para algumas bibliotecas (ex: LightGBM, XGBoost, OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1 \
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

# Criar diretórios para persistência e ajustar permissões para o usuário do Hugging Face (1000)
RUN mkdir -p mlruns data_lake models && chmod -R 777 mlruns data_lake models

# Hugging Face Spaces usa a porta 7860 por padrão
EXPOSE 7860

# O comando padrão para rodar o Streamlit na porta correta do Hugging Face
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
