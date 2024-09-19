import streamlit as st
from utils.assets_loader import set_assets, render_header, render_footer

def render_about_us():
    st.title("Conoce a Nuestro Equipo")

    team_members = [
        {"name": "Manuel Castro Villegas", "profession": "Ingeniero | Data Scientist",
         "github": "https://github.com/Manuelcastro97", "linkedin": "https://www.linkedin.com/in/manuelcastro97"},
        {"name": "Valentín Catalin Olalau", "profession": "S&OP | Demand Planner | Data Analyst",
         "github": "https://github.com/SrOlalau", "linkedin": "https://www.linkedin.com/in/valent%C3%ADn-catal%C3%ADn-olalau/"},
        {"name": "Camilo Cortés Gómez", "profession": "Data Scientist",
         "github": "https://github.com/cvmilo0", "linkedin": "https://www.linkedin.com/in/camilo-cortes-gomez"},
        {"name": "Diego Gloria Salamanca", "profession": "Administrador Financiero | Data Scientist",
         "github": "https://github.com/Gloriuss", "linkedin": "https://www.linkedin.com/in/diego-gloria-salamanca/"},
        {"name": "Álvaro Oñoro Moya", "profession": "Quality Manager",
         "github": "https://github.com/Ixelar", "linkedin": ""},
        {"name": "Alonso Valdés González", "profession": "Economista | Data Scientist",
         "github": "https://github.com/Alonsomar", "linkedin": "https://www.linkedin.com/in/alonso-vald%C3%A9s-gonz%C3%A1lez-b44535135/"}
    ]

    team_html = """
        <div class='team-container'>
        """
    # Construir los contenedores de cada miembro como parte del bloque HTML
    for member in team_members:
        team_html += f"""
        <div class='member-card'>
            <h2>{member['name']}</h2>
            <p>{member['profession']}</p>
            <p>
                <a href="{member['github']}" target="_blank" rel="noopener noreferrer"><i class="fab fa-github"></i></a>
                <a href="{member['linkedin']}" target="_blank" rel="noopener noreferrer"><i class="fab fa-linkedin"></i></a>
            </p>
        </div>
        """

    # Cerrar el contenedor principal
    team_html += "</div>"

    st.html(team_html)



if __name__ == "__main__":
    st.set_page_config(page_title="Nosotros", page_icon="👥", layout="wide")
    set_assets()
    render_about_us()
    render_footer()
