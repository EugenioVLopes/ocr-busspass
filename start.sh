#!/bin/sh

# Iniciar o Flask em segundo plano
python /app/app.py &

# Iniciar o Nginx
nginx -g 'daemon off;'
