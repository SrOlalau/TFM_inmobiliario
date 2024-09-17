import streamlit as st
import os

# Obtener el directorio base del script actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_custom_css():
    # Construir la ruta para el archivo CSS
    css_path = os.path.join(BASE_DIR, '..', 'static', 'main.css')
    
    # Abrir el archivo CSS y aplicarlo
    with open(css_path) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


def set_sidebar_logo():
    # Construir la ruta para las imágenes del logo
    logo_long_path = os.path.join(BASE_DIR, '..', 'static', 'images', 'logo_long.png')
    icon_image_path = os.path.join(BASE_DIR, '..', 'static', 'images', 'logo.png')
    
    # Usar las rutas para los logos
    st.sidebar.image(logo_long_path)
    st.sidebar.image(icon_image_path)


def set_logo():
    left_co, cent_co, last_co = st.columns(3)
    
    # Construir la ruta para la imagen del logo
    logo_long_path = os.path.join(BASE_DIR, '..', 'static', 'images', 'logo_long.png')
    
    with cent_co:
        st.image(logo_long_path, use_column_width=True)


def set_assets():
    load_custom_css()
    set_sidebar_logo()
    set_logo()


def render_header():
    st.markdown("""
        <div class="header">
            <div class="header-content">
                <h1 class="header-title">Spain Housing Predictor</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
        <div class="footer">
            <p>&copy; 2024 Precio Piso. Todos los derechos reservados.</p>
        </div>
    """, unsafe_allow_html=True)
