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


@st.cache_data # Zorgt ervoor dat de dataframe altijd geladen is
def data1():
    with open('pages/result_shivano.json', 'r') as file:
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



def predict_temperature(df, temp_column):
    X = df[['year']]
    y = df[temp_column]
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.arange(2024, 2051).reshape(-1, 1)
    predicted = model.predict(future_years)
    return future_years.flatten(), predicted

@st.cache_data
def data3():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # Define the API request parameters
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": [52.5244, 52.374, 50.8505],  # Berlin, Amsterdam, Brussels
        "longitude": [13.4105, 4.8897, 4.3488],  # Berlin, Amsterdam, Brussels
        "start_date": "1950-01-01",
        "end_date": "2050-12-31",
        "models": ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S"],
        "daily": ["temperature_2m_mean"]
    }
    
    # Fetch data from the API for each city
    return openmeteo.weather_api(url, params=params)
responses = data3()

# Create an empty list to store DataFrames
city_dfs = []

# Define city names
cities = ['Berlin', 'Amsterdam', 'Brussels']

# Process the response for each city
for idx, city in enumerate(cities):
    response = responses[idx]
    
    # Extract daily data for temperature
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    
    # Create a DataFrame from the data
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_mean": daily_temperature_2m_mean,
        "city": [city] * len(daily_temperature_2m_mean)
    }
    
    # Append the DataFrame for each city to the list
    city_dfs.append(pd.DataFrame(daily_data))

# Concatenate all city DataFrames into one
daily_dataframe = pd.concat(city_dfs, ignore_index=True)

# Convert the date column to datetime format
daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'])

# Filter data to only include June, July, and August
daily_dataframe['month'] = daily_dataframe['date'].dt.month
summer_data = daily_dataframe[daily_dataframe['month'].isin([6, 7, 8])]

# Drop the 'month' column
summer_data = summer_data.drop(columns=['month'])

august_data = summer_data[summer_data['date'].dt.month == 8]
august_data['year'] = august_data['date'].dt.year

# Assume august_data is already loaded with necessary fields
# Generate some example min/max temperature data for August (replace with actual data)
august_data['min_temperature_2m'] = august_data['temperature_2m_mean'] - 5  # Example for min temp
august_data['max_temperature_2m'] = august_data['temperature_2m_mean'] + 5  # Example for max temp

# Define cities
cities = august_data['city'].unique()

# Create a Plotly figure
fig = go.Figure()

# Define function for adding traces (min/max) and predictions
def add_temperature_traces(fig, city, visible_min=False, visible_max=False):
    city_august_data = august_data[(august_data['city'] == city) & (august_data['year'] <= 2024)]

    # Add historical data for min temperature
    fig.add_trace(go.Scatter(x=city_august_data['year'], y=city_august_data['min_temperature_2m'], mode='markers',
                             name=f'Historical Min Temp {city}', visible=visible_min))
    
    # Add historical data for max temperature
    fig.add_trace(go.Scatter(x=city_august_data['year'], y=city_august_data['max_temperature_2m'], mode='markers',
                             name=f'Historical Max Temp {city}', visible=visible_max))
    
    # Add prediction for min temperature (2024-2050)
    future_years, predicted_min = predict_temperature(city_august_data, 'min_temperature_2m')
    fig.add_trace(go.Scatter(x=future_years, y=predicted_min, mode='lines',
                             name=f'Predicted Min Temp {city} (2024-2050)', visible=visible_min))
    
    # Add prediction for max temperature (2024-2050)
    future_years, predicted_max = predict_temperature(city_august_data, 'max_temperature_2m')
    fig.add_trace(go.Scatter(x=future_years, y=predicted_max, mode='lines',
                             name=f'Predicted Max Temp {city} (2024-2050)', visible=visible_max))

# Initial visibility setup for each city and temperature type (Min Temp of Berlin by default)
for city in cities:
    add_temperature_traces(fig, city, visible_min=(city == 'Berlin'), visible_max=False)

# Streamlit Radio Buttons for City Selection (Single Selection)
selected_city = st.radio("Select City", options=cities, index=0)

# Streamlit Radio Buttons for Temperature Type Selection (Single Selection)
selected_temp_type = st.radio("Select Temperature Type", options=["Min Temp", "Max Temp"], index=0)

# Update the visibility of the traces based on the selected city and temperature type
for i, city in enumerate(cities):
    is_selected_city = (city == selected_city)
    for j in range(4):  # Each city has 4 traces: Min historical, Min predicted, Max historical, Max predicted
        fig.data[i * 4 + j].visible = (selected_temp_type == "Min Temp" and is_selected_city and j in [0, 2]) or \
                                      (selected_temp_type == "Max Temp" and is_selected_city and j in [1, 3])

# Update the layout
fig.update_layout(
    title=f"{selected_temp_type} Predictions for {selected_city} (2024-2050)",
    xaxis_title="Year",
    yaxis_title="Temperature (°C)"
)

# Show the figure in Streamlit
st.plotly_chart(fig)
