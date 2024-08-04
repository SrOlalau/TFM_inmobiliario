#!/bin/bash

# Ejecuta el script de Python una vez al iniciar el contenedor
python /app/main.py

# Luego inicia cron en primer plano
cron -f