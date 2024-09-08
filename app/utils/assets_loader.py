import streamlit as st

def load_custom_css():
    with open('./app/static/main.css') as f:
        css = f.read()
    st.html(f'<style>{css}</style>')


def set_sidebar_logo():
    st.logo(image="./app/static/images/logo_long.png", icon_image="./app/static/images/logo.png")


def set_logo():
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("./app/static/images/logo_long.png", use_column_width =True)


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
