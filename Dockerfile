# Usa uma imagem base oficial do Python
FROM python:3.9-slim-buster

# Define o diretório de trabalho no container
WORKDIR /app

# Instala dependências do poppler
RUN apt-get update && apt-get install -y libstdc++6

# Copia a pasta do Poppler para o container
COPY poppler-23.11.0 /app/poppler-23.11.0 

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação para o container
COPY . .

# Instala o Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-por

# Expõe a porta que a aplicação vai usar
EXPOSE 80

# Comando para iniciar a aplicação com Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]