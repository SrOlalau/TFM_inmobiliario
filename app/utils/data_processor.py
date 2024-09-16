import streamlit as st
import folium
from streamlit_folium import st_folium
import pgeocode
from geopy.geocoders import Nominatim
from functools import partial

geolocator = Nominatim(user_agent="precioPiso")
geocode = partial(geolocator.geocode, language="es")




# Diccionario para trabsformar el campo state_name en un valor valido de CCAA.
PGEO_CCAA = {
    'Andalucia': 'Andalucía',
    'Cataluna': 'Cataluña',
    'Cantabria': 'Cantabria',
    'Madrid': 'Comunidad de Madrid',
    'Comunidad Valenciana': 'Comunidad Valenciana',
    'Pais Vasco': 'País Vasco',
}

def valida_seleccion(pgeocode_resp=None, geopy_state=None):
    if pgeocode_resp is not None:
        if PGEO_CCAA.get(pgeocode_resp['state_name']):
            return PGEO_CCAA.get(pgeocode_resp['state_name'])
        else:
            st.error(f"Por favor, seleccione un punto en una comunidad autónoma válida: {PGEO_CCAA.values()}")
            return None
    if geopy_state is not None:
        if geopy_state in PGEO_CCAA.values():
            return geopy_state
        else:
            st.error(f"Por favor, seleccione un punto en una comunidad autónoma válida: {PGEO_CCAA.values()}")
            return None

def select_lat_lon():
    """
    Permite al usuario seleccionar una latitud y longitud utilizando un código postal o seleccionando en un mapa.

    Returns:
    Tuple[float, float] or None: Devuelve la latitud y longitud seleccionadas, o None si no se selecciona ninguna.
    """
    # Inicializa el objeto de búsqueda por código postal
    nomi = pgeocode.Nominatim('es')

    # Método de selección de ubicación
    location_method = st.radio("Seleccione el método de ubicación:", ("Código postal", "Seleccionar en el mapa"))

    if location_method == "Código postal":
        codigo_postal = st.text_input("Ingrese su código postal:")

        # Validación del código postal
        if codigo_postal and (not codigo_postal.isdigit() or len(codigo_postal) != 5):
            st.error("Por favor, ingrese un código postal válido de 5 dígitos.")
            return None
        elif codigo_postal:
            resp = nomi.query_postal_code(codigo_postal)
            if resp['country_code'] == 'ES':
                st.success(f"Código postal válido: {codigo_postal}")
                return resp['latitude'], resp['longitude'], valida_seleccion(pgeocode_resp=resp)
            else:
                st.error("Código postal no asociado a una ubicación válida en España.")
                return None
    else:

        if 'selected_point' not in st.session_state:
            st.session_state.selected_point = None

        if location_method == "Seleccionar en el mapa":
            st.write("Seleccione un punto en el mapa:")
            # Crear el mapa centrado en la ubicación deseada con la capa base Stadia Stamen Toner Background
            m = folium.Map(
                location=[40.5, -3.62],
                tiles='https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}{r}.png',
                attr='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://www.stamen.com/" target="_blank">Stamen Design</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                zoom_start=6,
                width='100%')

            m.add_child(folium.LatLngPopup())
            map_data = st_folium(m)

            if map_data.get("last_clicked"):
                st.session_state.selected_point = map_data['last_clicked']

        if st.session_state.selected_point:
            lat = st.session_state.selected_point['lat']
            lon = st.session_state.selected_point['lng']
            geopy_state = get_state_from_latlon(lat, lon)

            return lat, lon, valida_seleccion(geopy_state=geopy_state)

    return None

def get_state_from_latlon(lat,lon):
    # lat = "25.594095"
    # lon = "85.137566"

    reverse = partial(geolocator.reverse, language="es")
    location = reverse(f"{lat}, {lon}")
    address = location.raw['address']
    state = address.get('state')
    return state
