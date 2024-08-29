import streamlit as st
from utils.assets_loader import set_assets, render_header, render_footer


def render_details():
    st.title("Detalles de la Aplicación")

    st.markdown("""
        ### Bienvenido a Spain Housing Predictor
        Esta aplicación está diseñada para ayudarte a tomar decisiones informadas sobre el mercado inmobiliario en España.
    """)

    st.write("### Características Clave:")
    st.write("#### 1. Predicción de Precios")
    st.write("Introduce detalles sobre una propiedad y obtén predicciones instantáneas sobre su valor de compra y alquiler.")

    st.write("#### 2. Análisis Estadístico")
    st.write("Explora estadísticas detalladas y visualizaciones sobre las tendencias del mercado inmobiliario en diversas regiones.")

    st.write("#### 3. Datos en Tiempo Real")
    st.write("Nuestros modelos utilizan datos en tiempo real para garantizar la precisión y relevancia de las predicciones.")

    st.image("https://via.placeholder.com/800x400", caption="Análisis y Predicción")

if __name__ == "__main__":
    st.set_page_config(page_title="Detalles", page_icon="📃")
    set_assets()
    render_header()
    render_details()
    render_footer()

