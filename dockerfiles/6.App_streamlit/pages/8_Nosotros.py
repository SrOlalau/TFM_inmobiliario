import streamlit as st
from app.utils.assets_loader import set_assets, render_header, render_footer

def render_about_us():
    st.title("Conoce a Nuestro Equipo")

    team_members = [
        {"name": "Manuel Castro Villegas", "info": "Descripción breve del miembro 1", "profession": "Profesión",
         "github": "https://github.com/Manuelcastro97", "linkedin": "https://linkedin.com/in/miembro1"},
        {"name": "Valentín Catalin Olalau", "info": "Descripción breve del miembro 2", "profession": "Profesión",
         "github": "https://github.com/SrOlalau", "linkedin": "https://www.linkedin.com/in/valent%C3%ADn-catal%C3%ADn-olalau/"},
        {"name": "Iván Camilo Cortés Gómez", "info": "Descripción breve del miembro 3", "profession": "Profesión",
         "github": "https://github.com/cvmilo0", "linkedin": "https://linkedin.com/in/miembro3"},
        {"name": "Diego Gloria Salamanca", "info": "Descripción breve del miembro 4", "profession": "Profesión",
         "github": "https://github.com/Gloriuss", "linkedin": "https://www.linkedin.com/in/diego-gloria-salamanca/"},
        {"name": "Álvaro Oñoro Moya", "info": "Descripción breve del miembro 5", "profession": "Profesión",
         "github": "https://github.com/Ixelar", "linkedin": "https://linkedin.com/in/miembro5"},
        {"name": "Alonso Valdés González", "info": "Descripción breve del miembro 6", "profession": "Profesión",
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
            <p>{member['info']}</p>
            <p><strong>Profesión: </strong>{member['profession']}</p>
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