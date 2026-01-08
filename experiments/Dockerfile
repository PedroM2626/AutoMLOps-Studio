# Usar imagem base oficial de Python
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para Prophet e outras bibliotecas
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas o arquivo de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código e o arquivo .env
COPY . .

# Expor a porta que a FastAPI usará
EXPOSE 8000

# Comando para rodar a aplicação
# Nota: app_serving.py deve ser gerado antes ou durante o build
CMD ["python", "app_serving.py"]
