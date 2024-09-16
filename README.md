# TFM Inmobiliario - Desarrollo Local

Bienvenido al repositorio del **TFM Inmobiliario**. Esta rama `main` está diseñada para el desarrollo y prueba del proyecto en un entorno local antes de ser implementado en un servidor, como se detalla en la [rama server](https://github.com/SrOlalau/TFM_inmobiliario/tree/Servidor).

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un sistema de análisis de datos inmobiliarios utilizando técnicas avanzadas de ciencia de datos, análisis exploratorio de datos (EDA), ingeniería de características y modelado predictivo. El flujo de trabajo del proyecto está estructurado para facilitar la implementación y desarrollo continuo, permitiendo a los desarrolladores realizar pruebas de manera eficiente en sus entornos locales antes de desplegar los cambios en el servidor.

## Estructura del Proyecto

A continuación se describe la estructura del proyecto y la funcionalidad de cada una de sus carpetas:

- [app](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/app): Contiene la aplicación principal que sirve como interfaz de usuario para interactuar con los modelos de análisis de datos.
  
- [.streamlit](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/.streamlit): Archivos de configuración para la aplicación desarrollada con Streamlit, facilitando la personalización del frontend.

- [data](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/data): Almacena los conjuntos de datos utilizados en el proyecto, tanto crudos como procesados.

- [datamunging](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/datamunging): Scripts para limpieza y transformación de datos, preparando los conjuntos de datos para el análisis y modelado.

- [datatuning](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/datatuning): Contiene scripts para ajustar los modelos de datos y optimizar su rendimiento.

- [EDA](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/EDA): Scripts y notebooks para el Análisis Exploratorio de Datos (Exploratory Data Analysis - EDA), utilizados para descubrir patrones y relaciones en los datos.

- [machinelearning](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/machinelearning): Implementaciones de modelos de aprendizaje automático utilizados en el análisis de datos inmobiliarios.

- [web_scraping_scripts](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/web_scraping_scripts): Scripts utilizados para la extracción de datos de sitios web relevantes para el mercado inmobiliario.

- [.gitignore](https://github.com/SrOlalau/TFM_inmobiliario/blob/main/.gitignore): Archivo que especifica los archivos y directorios que Git debe ignorar en el control de versiones.

- [main.py](https://github.com/SrOlalau/TFM_inmobiliario/blob/main/main.py): Script principal para ejecutar la aplicación localmente.

- [requirements.txt](https://github.com/SrOlalau/TFM_inmobiliario/blob/main/requirements.txt): Archivo que contiene todas las dependencias de Python necesarias para ejecutar el proyecto.

## Flujo de Trabajo Principal

1. **Obtención de Datos**: Utiliza los scripts en la carpeta [web_scraping_scripts](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/web_scraping_scripts) para extraer datos de fuentes inmobiliarias. Los datos se almacenan en la carpeta [data](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/data).

2. **Limpieza y Transformación de Datos**: Los datos crudos se procesan utilizando scripts en la carpeta [datamunging](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/datamunging) para obtener un conjunto de datos limpio y listo para el análisis.

3. **Análisis Exploratorio de Datos (EDA)**: Se lleva a cabo un EDA utilizando los scripts y notebooks en [EDA](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/EDA) para identificar patrones, anomalías y relaciones significativas.

4. **Modelado Predictivo**: Se desarrollan y ajustan modelos de aprendizaje automático en la carpeta [machinelearning](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/machinelearning). Los modelos se entrenan, validan y optimizan utilizando scripts en [datatuning](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/datatuning).

5. **Implementación de la Aplicación**: La aplicación es desarrollada y probada localmente a través del script [main.py](https://github.com/SrOlalau/TFM_inmobiliario/blob/main/main.py), utilizando la configuración de Streamlit proporcionada en [.streamlit](https://github.com/SrOlalau/TFM_inmobiliario/tree/main/.streamlit).

6. **Despliegue al Servidor**: Una vez que las pruebas locales son satisfactorias, los cambios se integran en la [rama server](https://github.com/SrOlalau/TFM_inmobiliario/tree/server) para su despliegue en el servidor de producción.

## Requisitos
Python 3.12 o mayor
