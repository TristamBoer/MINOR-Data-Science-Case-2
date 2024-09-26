import streamlit as st
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.graph_objects as go
import plotly_express as pxe
import numpy as np
np.float_ = np.float64
from prophet import Prophet
import plotly_express as px

st.set_page_config(page_title='Historical Weather Data')

st.markdown('# Historical Weather Data')

@st.cache_data # Zorgt ervoor dat de dataframe altijd geladen is
def data():
	'''
	Alle code binnen de data() functie is gebruikt van de Openmeteo website zelf.
	'''
	
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)
	
	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": 52.374,
		"longitude": 4.8897,
		"start_date": "2000-01-01",
		"end_date": "2024-01-01",
		"daily": ["temperature_2m_mean", "apparent_temperature_mean", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"],
		"timezone": "Europe/Berlin"
	}
	responses = openmeteo.weather_api(url, params=params)
	
	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	
	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
	daily_apparent_temperature_mean = daily.Variables(1).ValuesAsNumpy()
	daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
	daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
	daily_rain_sum = daily.Variables(5).ValuesAsNumpy()
	daily_snowfall_sum = daily.Variables(6).ValuesAsNumpy()
	daily_precipitation_hours = daily.Variables(7).ValuesAsNumpy()
	daily_wind_speed_10m_max = daily.Variables(8).ValuesAsNumpy()
	daily_wind_gusts_10m_max = daily.Variables(9).ValuesAsNumpy()
	daily_wind_direction_10m_dominant = daily.Variables(10).ValuesAsNumpy()
	
	daily_data = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}
	
	daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
	daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
	daily_data["daylight_duration"] = daily_daylight_duration
	daily_data["sunshine_duration"] = daily_sunshine_duration
	daily_data["precipitation_sum"] = daily_precipitation_sum
	daily_data["rain_sum"] = daily_rain_sum
	daily_data["snowfall_sum"] = daily_snowfall_sum
	daily_data["precipitation_hours"] = daily_precipitation_hours
	daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
	daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
	daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
	
	daily_dataframe = pd.DataFrame(data=daily_data)
	daily_dataframe['date'] = daily_dataframe['date'].dt.date
	daily_dataframe.index = daily_dataframe['date']
	daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'], yearfirst=True)
	daily_dataframe['daylight_duration'] = (daily_dataframe['daylight_duration'] / 60) / 60
	daily_dataframe['sunshine_duration'] = (daily_dataframe['sunshine_duration'] / 60) / 60
	return daily_dataframe

daily_dataframe = data()

st.markdown(
	    '''  
	    Binnen de Historical Weather Data API worden de volgende onderwerpen op volgorde besproken:  
	    - **Uitleg DataFrame kolommen**  
	    - **Interactieve scatterplot visualisatie van DataFrame**   
	    - **Interactieve scatterplot op basis van windrichting**  
	    - **Voorspellend model voor gemiddelde temperatuur**  
	    '''
	)

st.header('Historical Weather DataFrame')
col1, col2 = st.columns([1,2.65])

with col1:
	st.markdown(
	    '''  
	    De Historical Weather DataFrame bevat de volgende kolommen:  
	    - **Date:**  
	      *Bevat de datums van alle datapunten. Elke rij in deze kolom is een nieuwe dag.*  
	    - **temperature_2m_mean:**  
	      *Bevat de gemiddelde temperatuur, in graden Celsius, gemeten op twee meter hoogte.*  
	    - **apparent_temperature_mean:**  
	      *Bevat de gemiddelde gevoelstemperatuur, in graden Celsius, gemeten op twee meter hoogte.*  
	    - **daylight_duration:**  
	      *Bevat hoelang, in uren, het op een dag daglicht is.*  
	    - **sunshine_duration:**  
	      *Bevat hoelang, in uren, de zon op een dag schijnt.*  
	    - **precipitation_sum:**  
	      *Bevat de som, in millimeters, van de hoeveelheid gevallen neerslag.*  
	    - **rain_sum:**  
	      *Bevat de som, in millimeters, van de hoeveelheid gevallen regen.*  
	    - **snowfall_sum:**  
	      *Bevat de som, in millimeters, van de hoeveelheid gevallen sneeuw.*  
	    - **precipitation_hours:**  
	      *Bevat hoelang, in uren, het op een dag heeft geregend.*  
	    - **wind_speed_10m_max:**  
	      *Bevat de maximale windsnelheid op een dag, in km/u.*  
	    - **wind_gusts_10m_max:**  
	      *Bevat de maximale windvlaag op een dag, in km/u.*  
	    - **wind_direction_10m_dominant:**  
	      *Bevat de dominante windrichting op een dag, in graden.*
	    '''
	)

with col2:
	st.dataframe(daily_dataframe, height=600)

st.text("") # Extra pagina ruimte tussen stukken 

st.header('Interactieve scatterplot van DataFrame')
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    variable1 = st.selectbox("Selecteer eerste variabel:", [
	'date', 'temperature_2m_mean', 'apparent_temperature_mean',
	'daylight_duration', 'sunshine_duration', 'precipitation_sum',
	'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max',
	'wind_gusts_10m_max', 'wind_direction_10m_dominant'
	])
			     
with col2:
    variable2 = st.selectbox("Selecteer tweede variabel:", [
	'date', 'temperature_2m_mean', 'apparent_temperature_mean',
	'daylight_duration', 'sunshine_duration', 'precipitation_sum',
	'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max',
	'wind_gusts_10m_max', 'wind_direction_10m_dominant'
	])


with col3:
    if variable1 and variable2:
        fig = px.scatter(daily_dataframe, x=variable1, y=variable2, title=f'{variable1} vs {variable2}')
        st.plotly_chart(fig)

st.text("") # Extra pagina ruimte tussen stukken 

def categorize_wind_direction(degrees):
    if (degrees >= 337.5) or (degrees < 22.5):
        return 'North'
    elif 22.5 <= degrees < 67.5:
        return 'North-East'
    elif 67.5 <= degrees < 112.5:
        return 'East'
    elif 112.5 <= degrees < 157.5:
        return 'South-East'
    elif 157.5 <= degrees < 202.5:
        return 'South'
    elif 202.5 <= degrees < 247.5:
        return 'South-West'
    elif 247.5 <= degrees < 292.5:
        return 'West'
    elif 292.5 <= degrees < 337.5:
        return 'North-West'

daily_dataframe['wind_direction_category'] = daily_dataframe['wind_direction_10m_dominant'].apply(categorize_wind_direction)

wind_direction_options = daily_dataframe['wind_direction_category'].unique()

st.header('Interactieve scatterplot windrichting')

col1, col2 = st.columns(2)

with col1:
	selected_wind_direction = st.selectbox('Selecteer windrichting:', options=wind_direction_options)

filtered_df = daily_dataframe[daily_dataframe['wind_direction_category'] == selected_wind_direction]

with col2:
	variable3 = st.selectbox("Selecteer y-as variabel:", [
	'temperature_2m_mean', 'apparent_temperature_mean',
	'sunshine_duration', 'daylight_duration', 'precipitation_sum',
	'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max',
	'wind_gusts_10m_max', 'wind_direction_10m_dominant'
	])

fig = px.scatter(data_frame=filtered_df,
                 x='date', y=variable3
		)

fig.update_layout(
    title=f'{variable3} gebaseerd op directie: {selected_wind_direction}',
    yaxis_title=f'{variable3}',
    xaxis_title='Date',
    title_font_size=18,
)

st.plotly_chart(fig)

st.text("") # Extra pagina ruimte tussen stukken 

st.header('Voorspellend model gemiddelde temperatuur')
@st.cache_resource
def prediction():
    forecast_data = daily_dataframe.rename(columns={'date': 'ds', 'temperature_2m_mean': 'y'})
    
    model = Prophet()
    model.fit(forecast_data)
    forecast = model.make_future_dataframe(periods=730)  # Voorspelling van twee jaar
    predictions = model.predict(forecast)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['y'],
                             mode='markers', name='Actual Data',
                             line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=predictions['ds'], y=predictions['yhat'],
                             mode='lines', name='Forecasted Data',
                             line=dict(color='green')))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=predictions['ds'], y=predictions['yhat_upper'],
        mode='lines', name='Upper Confidence Interval',
        line=dict(width=0), showlegend=False))
    
    fig.add_trace(go.Scatter(
        x=predictions['ds'], y=predictions['yhat_lower'],
        fill='tonexty', name='Confidence Interval',
        line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=True))
    
    return fig, predictions['ds'].min(), predictions['ds'].max()

fig, min_date, max_date = prediction()

# Datum Slider
start_date, end_date = st.slider('Select lengte datum:',
                                 min_value=min_date.date(), max_value=max_date.date(),
                                 value=(min_date.date(), max_date.date()), format="YYYY-MM-DD")

fig.update_layout(title='Temperature Forecast',
                  xaxis_title='Date',
                  yaxis_title='Temperature (°C)',
                  title_font_size=18)

fig.update_xaxes(range=[str(start_date), str(end_date)])

st.plotly_chart(fig)
