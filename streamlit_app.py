import streamlit as st

def page_config():
        st.set_page_config(layout='wide',
                          page_title="Openmeteo Weather App")
page_config()

st.sidebar.success("Selecteer een van de bovenstaande pagina's.")

st.title('Openmeteo & KNMI Weather app')

# st.info('This is a weather app')

st.write(
        '''
        Voor dit dashboard is gebruikt gemaakt van weerdata afkomstig van de OpenMeteo API. OpenMeteo bevat accurate realtime en historische weerdata. 
        De data van de API is gemakkkelijk te gebruiken, doordat een overzichtelijk webpagina bevat. Waarin de user kan kiezen welke variabelen ze willen gebruiken. 
        Daarnaast is de API volledig openbaar en vereist geen API-sleutel. De data is beschikbaar is in verschillende tijdsintervallen, zoals uur- en dagvoorspellingen.

        -  **https://open-meteo.com/**
        '''
)

st.write(
        '''
        Daarnaast biedt de KNMI (Koninklijk Nederlands Meteorologisch Instituut) uitgebreide weer een klimaatdata voor Nederland. 
        Het KNMI, net zoals de OpenMeteo API, levert zowel actuele data als historische datasets. 
        Net als de OpenMeteo API, is de KNMI dataset makkelijk aan te passen. De user kan zelf kiezen welke variabelen beschikbaar zijn.

        - **https://daggegevens.knmi.nl/klimatologie/daggegevens**
        '''
)

st.write(
        '''
        De combinatie van de Openmeteo API en KNMI-data zorgen voor mengsel van gedetailleerde lokale en globale gegevens.
        '''
)
