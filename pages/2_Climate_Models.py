import streamlit as st
import json
import requests_cache
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import openmeteo_requests
from retry_requests import retry
import streamlit as st
from sklearn.linear_model import LinearRegression

def page_config():
        st.set_page_config(layout='wide',
                          page_title="Climate Models")
page_config()

uploaded_file = st.file_uploader("result_shivano.json", type="json")

@st.cache_data # Zorgt ervoor dat de dataframe altijd geladen is
def data1():
    with open('result_shivano.json', 'r') as file:
        return json.load(file)

knmi_df = pd.DataFrame(data1())

# Convert the date column to datetime and ensure temperature (TX) is scaled
knmi_df['date'] = pd.to_datetime(knmi_df['date'])
knmi_df['TX'] = knmi_df['TX'] / 10  # If the temperatures are in tenths of degrees



@st.cache_data # Zorgt ervoor dat de dataframe altijd geladen is
def data2():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    openmeteo = openmeteo_requests.Client(session=cache_session)
    
    # API parameters for Open-Meteo to get max temperature data
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": 52.374,  # Amsterdam
        "longitude": 4.8897,
        "start_date": "1950-01-01",
        "end_date": "2050-12-31",
        "daily": ["temperature_2m_max"]
    }
    
    # Retrieve Open-Meteo data (Max Temperature for Amsterdam)
    meteo_responses = openmeteo.weather_api(url, params=params)
    meteo_daily = meteo_responses[0].Daily()
    
    # Extract temperature max from Open-Meteo
    meteo_temperature_max = meteo_daily.Variables(0).ValuesAsNumpy()
    
    # Create a DataFrame for the Open-Meteo data
    meteo_data = {
        "date": pd.date_range(
            start=pd.to_datetime(meteo_daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(meteo_daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=meteo_daily.Interval()),
            inclusive="left"
        ),
        "temperature_max_meteo": meteo_temperature_max
    }
    return pd.DataFrame(data=meteo_data)
meteo_df = data2()



# Merge the KNMI and Open-Meteo datasets on the date column
combined_df = pd.merge(knmi_df[['date', 'TX']], meteo_df, on='date', how='inner')

# Extract year from date for yearly averages
combined_df['year'] = combined_df['date'].dt.year

# Calculate yearly averages for both KNMI and Open-Meteo
yearly_avg_df = combined_df.groupby('year').agg(
    TX_avg=('TX', 'mean'),
    meteo_avg=('temperature_max_meteo', 'mean')
).reset_index()

# Calculate the average of both sources for each year
yearly_avg_df['temperature_avg'] = (yearly_avg_df['TX_avg'] + yearly_avg_df['meteo_avg']) / 2

# Train a linear regression model on the data
X = yearly_avg_df[['year']]  # Year as the feature
y = yearly_avg_df['temperature_avg']  # Average temperature as target

model = LinearRegression()
model.fit(X, y)

# Make predictions for future years (2024-2040)
future_years = pd.DataFrame({'year': np.arange(2024, 2041)})
future_predictions = model.predict(future_years)

# Streamlit checkboxes for toggling each trace
show_knmi = st.checkbox("Show KNMI Yearly Avg Max Temperature", value=True)
show_meteo = st.checkbox("Show Open-Meteo Yearly Avg Max Temperature", value=True)
show_avg = st.checkbox("Show Average Temperature", value=True)
show_predictions = st.checkbox("Show Prediction (2024-2040)", value=True)

# Create the plotly figure
fig = go.Figure()

# Add the KNMI yearly average trace (red)
fig.add_trace(go.Scatter(
    x=yearly_avg_df['year'], y=yearly_avg_df['TX_avg'], 
    mode='lines', name='KNMI Yearly Avg Max Temperature', 
    line=dict(color='red'), visible=show_knmi))

# Add the Open-Meteo yearly average trace (blue)
fig.add_trace(go.Scatter(
    x=yearly_avg_df['year'], y=yearly_avg_df['meteo_avg'], 
    mode='lines', name='Open-Meteo Yearly Avg Max Temperature', 
    line=dict(color='blue'), visible=show_meteo))

# Add the average trace (purple)
fig.add_trace(go.Scatter(
    x=yearly_avg_df['year'], y=yearly_avg_df['temperature_avg'], 
    mode='lines', name='Average Temperature', 
    line=dict(color='purple'), visible=show_avg))

# Add the prediction trace (green, dashed line)
fig.add_trace(go.Scatter(
    x=future_years['year'], y=future_predictions, 
    mode='lines', name='Prediction (2024-2040)', 
    line=dict(color='green', dash='dash'), visible=show_predictions))

# Update layout
fig.update_layout(
    title="Comparison of Yearly Average Max Temperatures and Predictions (2024-2040)",
    xaxis_title="Year",
    yaxis_title="Temperature (°C)"
)

# Display the plot
st.plotly_chart(fig)
