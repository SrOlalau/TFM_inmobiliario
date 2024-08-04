#!/bin/bash
set -e

# Ejecuta el script de Python
python -u /app/main.py &

# Ejecuta el cron en primer plano
cron -f
