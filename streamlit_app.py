import streamlit as st

def page_config():
        st.set_page_config(layout='wide',
                          page_title="Openmeteo Weather App")
page_config()

st.sidebar.success("Selecteer een van de bovenstaande pagina's.")

st.title('Openmeteo & KNMI Weather app')

st.info('Pagina bevat alleen informatie over de API, dataset en wat er wordt bespoken per onderwerp.')

st.markdown(
    '''
    Voor dit dashboard is gebruikt gemaakt van weerdata afkomstig van de OpenMeteo API. OpenMeteo bevat accurate realtime en historische weerdata.   
    De data van de API is gemakkelijk te gebruiken, doordat een overzichtelijke webpagina bevat. Waarin de user kan kiezen welke variabelen ze willen gebruiken.   
    Daarnaast is de API volledig openbaar en vereist geen API-sleutel. De data is beschikbaar in verschillende tijdsintervallen, zoals uur- en dagvoorspellingen.

    - **[OpenMeteo](https://open-meteo.com/)**

    Daarnaast biedt de KNMI (Koninklijk Nederlands Meteorologisch Instituut) uitgebreide weer- en klimaatdata voor Nederland.   
    Het KNMI, net zoals de OpenMeteo API, levert zowel actuele data als historische datasets.    
    Net als de OpenMeteo API, is de KNMI dataset makkelijk aan te passen. De user kan zelf kiezen welke variabelen beschikbaar zijn.  
    De KNMI bevat zelf ook een API, maar door een vereiste van een API-sleutel is deze niet gebruikt.   
    Hierdoor is een openbare dataset gebruikt.

    - **[KNMI](https://daggegevens.knmi.nl/klimatologie/daggegevens)**
    '''
)

st.write(
        '''
        De combinatie van de Openmeteo API en KNMI-data zorgen voor mengsel een van gedetailleerde lokale en globale gegevens.  
        Binnen deze dashboard is er gekeken naar twee verschillende onderdelen van de OpenMeteo API:  
        - **Historical Weather**  
        - **Climate Change**  
        '''
)

col1, col2 = st.columns(2)

with col1:
        st.header('Historical Weather')
        st.markdown(
                '''
                - **Uitleg DataFrame kolommen**  
                - **Interactieve scatterplot visualisatie van DataFrame**   
                - **Interactieve scatterplot op basis van windrichting**  
                - **Voorspellend model voor gemiddelde temperatuur**  
                '''
)

with col2:
        st.header('Climate Change')
        st.markdown(
        '''
        
        '''
)
