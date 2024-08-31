import streamlit as st
from utils.assets_loader import set_assets, render_header, render_footer


def render_content():
    st.title("Bienvenido a Vivo y Rento")
    st.subheader("Tu herramienta para predecir precios de viviendas en España")

    st.markdown(
        """
        ¡Gracias por visitar nuestra aplicación! Aquí podrás explorar estadísticas de precios, realizar predicciones de valor de inmuebles y encontrar ofertas vigentes en el mercado español.
        """
    )

    st.image("https://via.placeholder.com/1500x500", caption="Explora el mercado inmobiliario en España", use_column_width=True)

    st.write("### Características Principales")
    st.write("- Predicción de precios de compra y alquiler.")
    st.write("- Análisis estadístico detallado.")
    st.write("- Información actualizada sobre ofertas en el mercado.")

    st.write("### ¿Qué puedes hacer con esta app?")
    st.write("1. Obtener predicciones precisas de precios inmobiliarios.")
    st.write("2. Analizar tendencias del mercado en distintas regiones.")
    st.write("3. Buscar y comparar ofertas de propiedades.")


if __name__ == "__main__":
    st.set_page_config(page_title="Inicio", page_icon="👋")
    set_assets()
    render_header()
    render_content()
    render_footer()
