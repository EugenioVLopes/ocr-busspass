FROM python:3.10-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar apenas o requirements.txt para instalar dependências
COPY requirements.txt .

# Atualizar o sistema e instalar pacotes necessários
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalar dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos do projeto para o contêiner
COPY . .

# Variável de ambiente para o Tesseract OCR
ENV TESSERACT_CMD=/usr/bin/tesseract

# Expor a porta 5000 para acesso ao Flask
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]