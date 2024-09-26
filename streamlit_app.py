import streamlit as st

def wide_space_default():
        st.set_page_config(layout=“wide”)
wide_space_default()


st.set_page_config(
    page_title="Openmeteo Weather App",
)

st.sidebar.success("Select a demo above.")

st.title('Openmeteo Weather app')

st.info('This is a weather app')
