import streamlit as st
from app.utils.assets_loader import set_assets, render_header, render_footer

def render_stats():
    st.title("Estadísticas del Mercado Inmobiliario")
    st.markdown("""
        ### Estadísticas de Precios en Diferentes Regiones
        Explora gráficos y datos sobre las tendencias de precios en el mercado inmobiliario en distintas regiones de España.
    """)

    st.subheader("Ejemplo de Gráfica:")
    st.markdown("_(Gráfica de ejemplo placeholder)_")
    st.image("https://via.placeholder.com/800x400", caption="Tendencia de precios por región")

    st.subheader("Filtros:")
    st.write("Selecciona una región para ver estadísticas específicas.")
    region = st.selectbox("Seleccionar Región", ["Madrid", "Barcelona", "Valencia", "Sevilla"])

    st.write(f"Mostrando datos para: {region}")

if __name__ == "__main__":
    st.set_page_config(page_title="Estadísticas", page_icon="📊")
    set_assets()
    render_header()
    render_stats()
    render_footer()