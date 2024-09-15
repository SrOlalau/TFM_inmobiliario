# Proyecto de Análisis Inmobiliario con Machine Learning

## 1. Introducción

Este proyecto es parte de nuestro **Trabajo de Fin de Máster (TFM)** para el programa **Máster en Data Science, Big Data & Business Analytics 2023-2024**. Hemos trabajado en equipo para desarrollar un sistema que scrapee datos de múltiples sitios web de propiedades (venta/alquiler), los limpie, procese y aplique técnicas de Machine Learning para generar predicciones basadas en los datos recolectados. Además, hemos desarrollado una plataforma web usando **Streamlit** para visualizar y analizar los resultados de manera interactiva.

El proyecto es completamente replicable y escalable gracias a su estructura modular, basada en contenedores Docker, que permite la gestión independiente de cada fase del proceso. Esto facilita la integración de nuevos datos o la modificación de cualquier parte del pipeline de forma eficiente.

### Equipo de trabajo:

A continuación, se presenta el equipo de desarrollo que ha trabajado en este proyecto, ordenado alfabéticamente:

- **Manuel Castro Villegas**   
  [GitHub](https://github.com/Manuelcastro97) | [LinkedIn](https://www.linkedin.com/in/manuelcastro97/)

- **Iván Camilo Cortés Gómez**  
  [GitHub](https://github.com/cvmilo0) | [LinkedIn](https://www.linkedin.com/in/camilo-cortes-gomez/)

- **Diego Gloria Salamanca**  
  [GitHub](https://github.com/Gloriuss) | [LinkedIn](https://www.linkedin.com/in/diego-gloria-salamanca/)

- **Valentín Catalin Olalau**  
  [GitHub](https://github.com/SrOlalau) | [LinkedIn](https://www.linkedin.com/in/valent%C3%ADn-catal%C3%ADn-olalau/)

- **Álvaro Oñoro Moya**  
  [GitHub](https://github.com/Ixelar) | [LinkedIn](https://linkedin.com/in/miembro5)

- **Alonso Valdés González**  
  [GitHub](https://github.com/Alonsomar) | [LinkedIn](https://www.linkedin.com/in/alonso-vald%C3%A9s-gonz%C3%A1lez-b44535135/)

## 2. Requisitos

El proyecto requiere el uso de Docker para facilitar la ejecución de los distintos componentes del pipeline de scraping, procesamiento y modelado de datos. Cada fase está contenida en un Docker independiente y se almacena en bases de datos PostgreSQL, también gestionadas en contenedores Docker.

### Requisitos del sistema:
- **Docker** (última versión)
- **Docker Compose**

### Configuración de PostgreSQL

Cada paso del proceso tiene su propia base de datos, lo que permite modularizar el flujo de trabajo y facilitar el control de los datos. Solo es necesario inicializar el contenedor PostgreSQL, ya que los scripts del proyecto se encargan de crear las tablas necesarias si no existen.

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

## 3. Flujo de datos

El flujo de datos está cuidadosamente estructurado en varios pasos, con cada fase gestionada a través de un contenedor Docker independiente y conectada a su base de datos correspondiente. Esto garantiza un procesamiento modular y eficiente. Todos los scripts y Dockerfiles están organizados en la carpeta `dockerfiles`, mientras que los archivos `docker-compose` necesarios para la ejecución están en la carpeta `Dockers_compose`.

### 3.1 Scraping de datos

El primer paso del proyecto es el scraping de sitios web de propiedades (alquiler y venta). Los scripts se encuentran en las carpetas `0.1_Scraping_pisos` y `0.2_Scraping_trovit`, y se ejecutan dentro de contenedores Docker, almacenando los datos en las bases de datos `scraping_pisos` y `scraping_trovit` respectivamente.

- **Ruta docker compose Pisos:** `Dockers_compose/0.1_Scraping_pisos.yaml`
- **Ruta docker compose Trovit:** `Dockers_compose/0.2_Scraping_trovit.yaml`

### 3.2 Scraping de puntos de interés (POI)

El siguiente paso consiste en obtener información geográfica sobre puntos de interés (POI) cercanos a las propiedades, como colegios, hospitales o transporte público, utilizando OpenStreetMaps. Los datos se almacenan en la base de datos `scraping_openstreetmaps`.

- **Ruta docker compose:** `Dockers_compose/0.3_Scraping_OpenStreetMaps.yaml`

### 3.3 Limpieza y procesamiento de datos (Data munging)

Aquí se procesan y limpian los datos obtenidos para eliminar duplicados, corregir valores y asegurarse de que están listos para el análisis. Los resultados se almacenan en la base de datos `datos_limpios`.

- **Ruta docker compose:** `Dockers_compose/1.Data_munging.yaml`

### 3.4 Añadir geo localizaciones

Se añade la información de las coordenadas geográficas de las propiedades para complementar el análisis. Esta etapa se ejecuta en un contenedor Docker y se almacena en la base de datos `geoloc`.

- **Ruta docker compose:** `Dockers_compose/2.Añadir_geo_localizaciones.yaml`

### 3.5 Ingeniería de variables

En este paso se integran los datos de geolocalización y los POI para enriquecer el dataset con variables adicionales que describen las cercanías a puntos clave. Los datos enriquecidos se almacenan en la base de datos `geo_y_poi`.

- **Ruta docker compose:** `Dockers_compose/3.Ingenieria_de_variables.yaml`

### 3.6 Data tuning

Se ajustan y optimizan las variables seleccionadas para mejorar el modelo de Machine Learning. Los resultados se almacenan en la base de datos `datatuning`.

- **Ruta docker compose:** `Dockers_compose/4.Data_tunning.yaml`

### 3.7 Machine learning

Se aplica un modelo de **Random Forest** para predecir los precios de las propiedades, basado en las variables optimizadas. El modelo entrenado se almacena en formato `.pickle` para su uso en la aplicación Streamlit.

- **Ruta docker compose:** `Dockers_compose/5.Machine_learning.yaml`

### 3.8 Aplicación Streamlit

La visualización interactiva de los resultados se realiza a través de una plataforma web desarrollada en **Streamlit**, donde se pueden explorar los datos procesados y los resultados del modelo.

- **Ruta docker compose:** `Dockers_compose/6.App_streamlit.yaml`

## 4. Conclusiones

Este proyecto demuestra cómo construir un pipeline completo de scraping, procesamiento de datos y Machine Learning de manera modular y escalable, usando contenedores Docker. La separación por etapas permite un control detallado sobre cada fase del proceso, facilitando la replicación y la adaptación a nuevos requisitos.

La plataforma Streamlit hace que sea fácil interactuar con los datos y obtener insights valiosos de manera rápida y eficiente. Si estás interesado en replicar este proyecto, sigue las instrucciones y ejecuta los scripts en el orden adecuado.

¡Esperamos que disfrutes el proyecto!

Visita nuestra plataforma en el siguiente enlace:

[Accede a nuestra web](http://preciopiso.com/)
