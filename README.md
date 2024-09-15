# Proyecto de Análisis Inmobiliario con Machine Learning

## 1. Introducción

Somos un equipo de entusiastas de la ciencia de datos y el desarrollo de software. Este proyecto tiene como objetivo crear un sistema que scrapee datos de múltiples sitios web de propiedades (venta/alquiler), los limpie, procese y aplique técnicas de Machine Learning para generar predicciones basadas en los datos recolectados. Además, hemos desarrollado una plataforma web usando **Streamlit** para visualizar y analizar los resultados de forma interactiva.

Este proyecto se puede replicar y escalar fácilmente gracias a su estructura modular y el uso de contenedores Docker, que permiten la separación y gestión independiente de cada etapa del proceso.

## 2. Requisitos

El proyecto requiere el uso de Docker para levantar varios contenedores que facilitan el scraping, el procesamiento de datos, y el almacenamiento en bases de datos independientes para cada paso del flujo de trabajo.

### Requisitos del sistema:
- **Docker** (última versión)
- **Docker Compose**

### Configuración de PostgreSQL

Cada paso del proyecto tiene su propia base de datos independiente para mantener los datos bien estructurados y facilitar la replicación del flujo de trabajo. Solo es necesario crear el contenedor de PostgreSQL, pues el código se encargará de crear las tablas si no existen y de gestionar correctamente las columnas con su tipo de dato correspondiente.

- **Base de datos para scraping de pisos:**
  - DB_NAME: `scraping_pisos`
  - DB_USER: `pisos`
  - DB_PASSWORD: `pisos`
  - DB_HOST: `10.1.2.2`
  - DB_PORT: `5437`
  - DB_Table_name: `scraping_pisos_tabla`

- **Base de datos para scraping de Trovit:**
  - DB_NAME: `scraping_trovit`
  - DB_USER: `trovit`
  - DB_PASSWORD: `trovit`
  - DB_HOST: `10.1.2.2`
  - DB_PORT: `5434`
  - DB_Table_name: `scraping_trovit_tabla`

- **Base de datos para scraping de OpenStreetMaps:**
  - DB_NAME: `scraping_openstreetmaps`
  - DB_USER: `POI`
  - DB_PASSWORD: `POI`
  - DB_HOST: `10.1.2.2`
  - DB_PORT: `5438`
  - DB_Table_name: `points_of_interest`

- **Base de datos para el data munging:**
  - DB_NAME: `datos_limpios`
  - DB_USER: `datos_limpios`
  - DB_PASSWORD: `datos_limpios`
  - DB_HOST: `10.1.2.2`
  - DB_PORT: `5439`
  - DB_Table_name: `consolidated_data`

- **Base de datos para añadir geo localizaciones:**
  - DB_NAME: `geoloc`
  - DB_USER: `geoloc`
  - DB_PASSWORD: `geoloc`
  - DB_HOST: `10.1.2.2`
  - DB_PORT: `5441`
  - DB_Table_name: `datos_limpios_con_geo`

- **Base de datos para la ingeniería de variables:**
  - DB_NAME: `geo_y_poi`
  - DB_USER: `geo_y_poi`
  - DB_PASSWORD: `geo_y_poi`
  - DB_HOST: `10.1.2.2`
  - DB_PORT: `5442`
  - DB_Table_name: `datos_limpios_con_geo_y_poi`

- **Base de datos para el data tuning y la aplicación Streamlit:**
  - DB_NAME: `datatuning`
  - DB_USER: `datatuning`
  - DB_PASSWORD: `datatuning`
  - DB_HOST: `10.1.2.2`
  - DB_PORT: `5444`
  - DB_Table_name: `Datos_finales`

Puedes lanzar estos contenedores y configuraciones utilizando Docker Compose y los archivos correspondientes dentro del proyecto.

## 3. Flujo de datos

El flujo de datos está cuidadosamente organizado en varios pasos, cada uno de ellos gestionado a través de un contenedor Docker específico, y conectado a una base de datos independiente para garantizar un procesamiento modular y eficiente. Todos los scripts, así como los Dockerfiles respectivos para construir los contenedores Docker, están en la carpeta `dockerfiles` organizados según el flujo de trabajo. Además, en la carpeta `Dockers_compose` tenemos los archivos `docker-compose` de cada parte del proyecto, para su ejecución de forma sencilla.

### 3.1 Scraping de datos

El primer paso del proyecto es el scraping de sitios web de propiedades (alquiler y venta). Los scripts se encuentran en las carpetas `0.1_Scraping_pisos` y `0.2_Scraping_trovit`, y se ejecutan dentro de un contenedor Docker que extrae los datos y los almacena en las bases de datos `scraping_pisos` y `scraping_trovit` respectivamente.

- **Ruta docker compose Pisos:** `Dockers_compose/0.1_Scraping_pisos.yaml`
- **Ruta docker compose Trovit:** `Dockers_compose/0.2_Scraping_trovit.yaml`

### 3.2 Scraping de puntos de interés (POI)

Después del scraping de los datos de alquiler y venta, el siguiente paso es scrapear datos de puntos de interés (POI). El script se encuentra en la carpeta `0.3_Scraping_OpenStreetMaps` y se ejecuta dentro de un contenedor Docker que extrae los datos y los almacena en la base de datos `scraping_openstreetmaps`.

- **Ruta docker compose:** `Dockers_compose/0.3_Scraping_OpenStreetMaps.yaml`

### 3.3 Limpieza y procesamiento de datos (Data munging)

Una vez que se tienen los datos base y los POI, se procede con la limpieza y transformación de los datos. En este paso, los datos se procesan y se dejan listos para el análisis. El script se encuentra en la carpeta `1.Data_munging` y se ejecuta dentro de un contenedor Docker que almacena los datos en la base de datos `datos_limpios`.

- **Ruta docker compose:** `Dockers_compose/1.Data_munging.yaml`

### 3.4 Añadir geo localizaciones

Durante este paso, se utilizan las coordenadas geográficas obtenidas previamente para mejorar los datos de las propiedades, añadiendo columnas adicionales con información de latitud y longitud. El script se encuentra en la carpeta `2.Añadir_geo_localizaciones` y se ejecuta dentro de un contenedor Docker que almacena los datos en la base de datos `geoloc`.

- **Ruta docker compose:** `Dockers_compose/2.Añadir_geo_localizaciones.yaml`

### 3.5 Ingeniería de variables

Una vez que los datos de las propiedades tienen las coordenadas geográficas, se integran con los datos de POI para añadir información relevante sobre los lugares cercanos. El script se encuentra en la carpeta `3.Ingenieria_de_variables` y se ejecuta dentro de un contenedor Docker que almacena los datos en la base de datos `geo_y_poi`.

- **Ruta docker compose:** `Dockers_compose/3.Ingenieria_de_variables.yaml`

### 3.6 Data tuning

Este paso involucra el ajuste y optimización de los datos para mejorar la calidad del modelo de Machine Learning, realizando ajustes en las variables seleccionadas. El script se encuentra en la carpeta `4.Data_tunning` y se ejecuta dentro de un contenedor Docker que almacena los datos en la base de datos `datatuning`.

- **Ruta docker compose:** `Dockers_compose/4.Data_tunning.yaml`

### 3.7 Machine learning

El último paso del proceso consiste en aplicar un modelo de Machine Learning (**Random Forest**) a los datos optimizados para predecir los precios de las propiedades basándose en las características extraídas. Este modelo se ejecuta dentro de un contenedor Docker y los resultados se almacenan en formato `.pickle` para que la aplicación Streamlit los utilice.

- **Ruta docker compose:** `Dockers_compose/5.Machine_learning.yaml`

### 3.8 Aplicación Streamlit

Este paso finaliza con la presentación de los resultados en una plataforma web interactiva usando **Streamlit**, donde los usuarios pueden consultar y analizar los datos procesados y los resultados del modelo de Machine Learning.

- **Ruta docker compose:** `Dockers_compose/6.App_streamlit.yaml`

## 4. Conclusiones

Este proyecto muestra cómo se puede construir un pipeline completo de scraping, procesamiento de datos y Machine Learning de manera modular y escalable usando Docker. La separación de bases de datos y contenedores por etapas permite un control detallado sobre el flujo de datos, facilitando la réplica del proceso en otros contextos o la adaptación del proyecto a nuevos requisitos.

Además, la plataforma Streamlit desarrollada para visualizar los resultados hace que sea fácil para cualquier usuario interactuar con los datos y obtener insights de manera rápida y eficiente. Si te interesa replicar este proyecto, asegúrate de seguir las instrucciones de configuración y ejecutar los scripts en el orden adecuado.

¡Esperamos que te sirva y que lo disfrutes!

Puedes visitar nuestra plataforma en el siguiente enlace:

[Accede a nuestra web](http://preciopiso.com/)

