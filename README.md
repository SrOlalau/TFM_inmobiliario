# Proyecto TFM

Este proyecto realiza web scraping de anuncios de propiedades inmobiliarias y guarda los datos en archivos CSV. Luego limpia los datos, entrena un modelo y finalmente despliega una app.
El proyecto está dividido en varias etapas y se organiza en diferentes scripts de Python.

## Estructura del Proyecto (por ahora)

```
Proyecto/
├── .venv/                    # Entorno virtual
├── web_scraping_scripts/     # Scripts de scraping
│   ├── __init__.py
│   ├── web_scraping_pisos.py
├── data/                     # Carpeta para almacenar datos descargados
│   └── RawDataTFM/
├── main.py                   # Script principal para ejecutar el scraping
├── .gitignore                # Archivo para ignorar ciertos archivos y carpetas en Git
└── README.md                 # Este archivo
```

## Requisitos

1. Python 3.6 o superior
2. pip (para instalar dependencias)
3. Git (para clonación y manejo de versiones)

## Configuración Inicial

1. **Clonar el repositorio:**

   ```sh
   git clone https://github.com/SrOlalau/TFM_inmobiliario
   cd Proyecto
   ```

2. **Crear y activar un entorno virtual:**

   ```sh
   python -m venv .venv
   source .venv/bin/activate   # En Windows, usa: .venv\Scripts\activate
   ```

3. **Instalar las dependencias:**

   ```sh
   pip install -r requirements.txt
   ```

## Ejecución del Proyecto

1. Asegúrate de que el entorno virtual esté activado.
2. Ejecuta el script principal:

   ```sh
   python main.py
   ```

## Estructura de Archivos y Directorios

- **web_scraping_scripts/**: Contiene los scripts de scraping. `web_scraping_pisos.py` incluye las funciones necesarias para realizar el scraping y guardar los datos.
- **data/**: Este directorio se utiliza para almacenar los datos descargados. No debe ser incluido en el control de versiones, por lo que está listado en `.gitignore`.
- **main.py**: Script principal que ejecuta el proceso de scraping.
- **.gitignore**: Archivo de configuración para excluir ciertos archivos y directorios del control de versiones.

## Cuidados y Buenas Prácticas

1. **Estructura del Proyecto**: Mantén la estructura del proyecto organizada. No muevas archivos o carpetas sin actualizar las rutas en los scripts correspondientes.
2. **Archivos Ignorados**: La carpeta `data` está incluida en `.gitignore` para evitar que los archivos de datos grandes sean rastreados por Git. No elimines esta entrada de `.gitignore`.
3. **Entorno Virtual**: Siempre utiliza un entorno virtual para instalar las dependencias del proyecto. Esto asegura que las dependencias estén aisladas y no interfieran con otros proyectos.
4. **Confirmación de Cambios**: Siempre confirma tus cambios con mensajes descriptivos. Esto facilita la revisión de cambios y la colaboración en el proyecto.
   ```sh
   git add .
   git commit -m "Descripción de los cambios realizados"
   git push origin <nombre-de-tu-rama>
   ```
5. **Colaboración**: Utiliza ramas para desarrollar nuevas características o hacer cambios. Esto facilita la revisión de código y evita conflictos en la rama principal (`main` o `master`).
6. **Creación de subcarpetas con scripts**: Utiliza nombres del tipo `snake_case` para nombrar carpetas y scripts. Para dejar disponibles los scripts de una carpeta en el entrno general, genera un archivo vacío de nombre `__init__.py`.

## Contribución

Para contribuir al proyecto, sigue estos pasos:

1. **Crear una rama nueva para tu característica o arreglo:**

   ```sh
   git checkout -b nombre-de-tu-rama
   ```

2. **Realizar tus cambios y confirmar:**

   ```sh
   git add .
   git commit -m "Descripción de los cambios"
   ```

3. **Empujar tu rama al repositorio remoto:**

   ```sh
   git push origin nombre-de-tu-rama
   ```

4. **Crear un Pull Request** en GitHub y espera la revisión de tu código.

## Recursos Adicionales

- [Documentación de GitHub](https://docs.github.com/)
- [Documentación de Python](https://docs.python.org/3/)

Si tienes alguna pregunta o necesitas ayuda, no dudes en contactarme.

---
¡GANEMOS!
