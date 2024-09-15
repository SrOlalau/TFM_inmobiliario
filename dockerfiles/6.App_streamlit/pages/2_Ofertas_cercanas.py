import streamlit as st
import folium as fl
from streamlit_folium import st_folium
import pgeocode
from app.utils.assets_loader import set_assets, render_header, render_footer

# Inicializa el objeto de búsqueda por código postal
nomi = pgeocode.Nominatim('es')

def geo_resp_to_text(res):
    address = f"{res['place_name']}, {res['county_name']}, {res['state_name']} ({res['latitude']}, {res['longitude']})"
    return address

def geo_search(codigo_postal):
    res = nomi.query_postal_code([str(codigo_postal)])
    return res.iloc[0]

def validate_codigo_postal():
    codigo_postal = st.text_input("Ingrese su código postal:")

    if codigo_postal and (not codigo_postal.isdigit() or len(codigo_postal) != 5):
        st.error("Ingrese un número válido")
        return None
    else:
        if codigo_postal:
            resp = geo_search(codigo_postal)
            if resp['country_code'] == 'ES':
                st.success(f"Código postal válido: {codigo_postal}, ubicación {geo_resp_to_text(resp)}")
                lat = resp['latitude']
                lon = resp['longitude']
                return codigo_postal, lat, lon
            else:
                st.error(f"Código postal no asociado a ubicación, intente con uno válido: {codigo_postal}")
                return None


def select_location_on_map():
    st.write("Seleccione un punto en el mapa:")
    m = fl.Map(location=[40.4168, -3.7038], zoom_start=6)
    m.add_child(fl.LatLngPopup())
    map_data = st_folium(m, height=300, width=600)

    if map_data.get("last_clicked"):
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        st.success(f"Ubicación seleccionada: ({lat}, {lon})")
        return lat, lon
    else:
        st.warning("Por favor, seleccione un punto en el mapa.")
        return None

def render_stats():
    st.title("Estima un precio justo para tu vivienda")

    # Opción de selección para método de ingreso de ubicación
    option = st.radio("Ingresar ubicación con:", ("Código postal", "Seleccionar un punto en el mapa"))

    if option == "Código postal":
        codigo_postal_val = validate_codigo_postal()
        if codigo_postal_val:
            codigo_postal, lat, lon = codigo_postal_val
            full_address = geo_search(codigo_postal)
            st.text(f"Ubicación: {full_address}")
    elif option == "Seleccionar un punto en el mapa":
        location = select_location_on_map()
        if location:
            lat, lon = location
            st.text(f"Coordenadas seleccionadas: ({lat}, {lon})")

    st.markdown("_(Aquí se mostrarían los resultados de la predicción cuando el modelo esté integrado)_")

if __name__ == "__main__":
    st.set_page_config(page_title="Estima precios", page_icon="🗺️", layout="wide")
    set_assets()
    render_stats()
    render_footer()