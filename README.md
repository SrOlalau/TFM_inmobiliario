# Proyecto de Análisis Inmobiliario con Machine Learning

## 1. Introducción

Somos un equipo de entusiastas de la ciencia de datos y el desarrollo de software. Este proyecto tiene como objetivo crear un sistema que scrapee datos de múltiples sitios web de propiedades (venta/alquiler), los limpie, procese y aplique técnicas de Machine Learning para generar predicciones basadas en los datos recolectados. Además, hemos desarrollado una plataforma web usando Streamlit para visualizar y analizar los resultados de forma interactiva.

Este proyecto se puede replicar y escalar fácilmente gracias a su estructura modular y el uso de contenedores Docker, que permiten la separación y gestión independiente de cada etapa del proceso.

## 2. Requisitos

El proyecto requiere el uso de Docker para levantar varios contenedores que facilitan el scraping, el procesamiento de datos, y el almacenamiento en bases de datos independientes para cada paso del flujo de trabajo.

### Requisitos del sistema:
- Docker (última versión)
- Docker Compose

### Configuración de PostgreSQL

Cada paso del proyecto tiene su propia base de datos independiente para mantener los datos bien estructurados y facilitar la replicación del flujo de trabajo y solo es necesario crear el docker de postgresql, pues el codigo en si se encargara de crear la tabla si no existe y de crear tambien todas las columnas con su tipo de dato correcto.

- Base de datos para scraping de pisos:
  - DB_NAME = `scraping_pisos`
  - DB_USER = `pisos`
  - DB_PASSWORD = `pisos`
  - DB_HOST = `10.1.2.2`
  - DB_PORT = `5437`
  - DB_Table_name = `scraping_pisos_tabla`

- Base de datos para el scraping de trovit
  - DB_NAME = `scraping_trovit`
  - DB_USER = `trovit`
  - DB_PASSWORD = `trovit`
  - DB_HOST = `10.1.2.2`
  - DB_PORT = `5434`
  - DB_Table_name = `scraping_trovit_tabla`

- Base de datos para el scraping de OpenStreetMaps
  - DB_NAME = `scraping_openstreetmaps`
  - DB_USER = `POI`
  - DB_PASSWORD = `POI`
  - DB_HOST = `10.1.2.2`
  - DB_PORT = `5438`
  - DB_Table_name = `points_of_interest`
- Base de datos para el data munging:
  - DB_NAME = `datos_limpios`
  - DB_USER = `datos_limpios`
  - DB_PASSWORD = `datos_limpios`
  - DB_HOST = `10.1.2.2`
  - DB_PORT = `5439`
  - DB_Table_name = `consolidated_data`

- Base de datos para añadir las geo localizaciones:
  - DB_NAME = `geoloc`
  - DB_USER = `geoloc`
  - DB_PASSWORD = `geoloc`
  - DB_HOST = `10.1.2.2`
  - DB_PORT = `5441`
  - DB_Table_name = `datos_limpios_con_geo`

- Base de datos para la ingenieria de variables:
  - DB_NAME = `geo_y_poi`
  - DB_USER = `geo_y_poi`
  - DB_PASSWORD = `geo_y_poi`
  - DB_HOST = `10.1.2.2`
  - DB_PORT = `5442`
  - DB_Table_name = `datos_limpios_con_geo_y_poi`

- Base de datos para el datatuning y de la cual se alimenta la aplicacion Streamlit:
  - DB_NAME = `datatuning`
  - DB_USER = `datatuning`
  - DB_PASSWORD = `datatuning`
  - DB_HOST = `10.1.2.2`
  - DB_PORT = `5444`
  - DB_Table_name = `Datos_finales`


Puedes lanzar estos contenedores y configuraciones utilizando Docker Compose y los archivos correspondientes dentro del proyecto.

## 3. Flujo de datos

El flujo de datos está cuidadosamente organizado en varios pasos, cada uno de ellos gestionado a través de un contenedor Docker específico, y conectado a una base de datos independiente para garantizar un procesamiento modular y eficiente.

### 3.1 Scraping de datos

El primer paso del proyecto es el scraping de sitios web de propiedades (alquiler y venta). Los scripts se encuentran en la carpeta `0.1_Scraping_pisos` y `0.2_Scraping_trovit`, y se ejecutan dentro de un contenedor Docker que extrae los datos y los almacena en la base de datos `scraping_pisos` y `scraping_trovit` de forma respectiva.

### 3.2 Scraping de puntos de interés (POI)

Después del scraping, el siguiente paso es scrapear datos de puntos de interés (POI) cercanos a las propiedades utilizando APIs externas como OpenStreetMaps. Este proceso se maneja en el contenedor `dockerfile_geoloc` y se almacena en la base de datos `geoloc_db`.

- **Script principal:** `main.py`
- **Base de datos:** `geoloc_db`
- **Contenedor Docker:** `dockerfile_geoloc`

### 3.3 Limpieza y procesamiento de datos (Data munging)

Una vez que se tienen los datos base y los POI, se procede con la limpieza y transformación de los datos. En este paso, los datos se procesan y se dejan listos para el análisis. Esto se gestiona en el contenedor `dockerfile_limpieza_datos`, y los datos procesados se almacenan en `variables_db`.

- **Script principal:** `main.py`
- **Base de datos:** `variables_db`
- **Contenedor Docker:** `dockerfile_limpieza_datos`

### 3.4 Geo localizaciones

Durante este paso, se utilizan las coordenadas geográficas obtenidas previamente para mejorar los datos de las propiedades, añadiendo columnas adicionales con información de latitud y longitud para facilitar la integración de los POI.

- **Script principal:** `main.py`
- **Base de datos:** `geoloc_db`
- **Contenedor Docker:** `dockerfile_geoloc`

### 3.5 Añadir los POI

Una vez que los datos de las propiedades tienen las coordenadas geográficas, se integran con los datos de POI para añadir información relevante de los lugares cercanos. Estos nuevos datos son almacenados nuevamente en `variables_db`.

- **Script principal:** `main.py`
- **Base de datos:** `variables_db`
- **Contenedor Docker:** `dockerfile_poi`

### 3.6 Data tuning

Este paso involucra el ajuste y optimización de los datos para mejorar la calidad del modelo de machine learning. Se realizan ajustes en las variables seleccionadas y se almacenan en `variables_db`.

- **Script principal:** `main.py`
- **Base de datos:** `variables_db`
- **Contenedor Docker:** `dockerfile_datatuning`

### 3.7 Machine learning

El último paso del proceso consiste en aplicar un modelo de machine learning (Random Forest) a los datos optimizados para predecir precios de las propiedades en base a las características extraídas. Este modelo se ejecuta dentro de un contenedor específico y los resultados son almacenados en la base de datos `ml_db`.

- **Script principal:** `main.py`
- **Base de datos:** `ml_db`
- **Contenedor Docker:** `dockerfile_machine_learning`

## 4. Conclusiones

Este proyecto muestra cómo se puede construir un pipeline completo de scraping, procesamiento de datos y machine learning de manera modular y escalable usando Docker. La separación de bases de datos y contenedores por etapas permite un control detallado sobre el flujo de datos, facilitando la réplica del proceso en otros contextos o la adaptación del proyecto a nuevos requisitos.

Además, la plataforma Streamlit desarrollada para visualizar los resultados hace que sea fácil para cualquier usuario interactuar con los datos y obtener insights de manera rápida y eficiente. Si te interesa replicar este proyecto, asegúrate de seguir las instrucciones de configuración y ejecutar los scripts en el orden adecuado.

¡Esperamos que te sirva y que lo disfrutes!
