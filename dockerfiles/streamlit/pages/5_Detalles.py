import streamlit as st
from app.utils.assets_loader import set_assets, render_header, render_footer
from app.utils.texts import licencias_text

def render_offers():
    st.title("Ofertas Actuales en el Mercado")
    st.markdown("""
        ### Descubre las últimas ofertas de propiedades en venta y alquiler.
        Esta sección muestra una lista de propiedades actuales en el mercado español.
    """)

    st.subheader("Ejemplo de Ofertas:")
    st.write("1. **Apartamento en Madrid** - €350,000")
    st.write("2. **Casa en Barcelona** - €500,000")
    st.write("3. **Ático en Valencia** - €450,000")

    st.markdown("_(Esta sección se conectará a una base de datos en una fase posterior para mostrar ofertas reales)_")

if __name__ == "__main__":
    st.set_page_config(page_title="Ofertas", page_icon="🔍", layout="wide")
    set_assets()
    render_offers()
    licencias_text()
    render_footer()