import streamlit as st


def load_custom_css():
    with open('./static/main.css') as f:
        css = f.read()
    st.html(f'<style>{css}</style>')


def set_sidebar_logo():
    st.logo(image="./static/images/logo_long.png", icon_image="./static/images/logo.png")


def set_logo():
    st.image("./static/images/logo_long.png", use_column_width=True)


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
            <p>&copy; 2024 Spain Housing Predictor. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)
