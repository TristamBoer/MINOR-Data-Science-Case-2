import streamlit as st

def page_config():
        st.set_page_config(layout='wide',
                          page_title="Openmeteo Weather App")
page_config()


st.sidebar.success("Select a demo above.")

st.title('Openmeteo Weather app')

st.info('This is a weather app')
