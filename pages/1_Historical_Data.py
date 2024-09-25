import streamlit as st
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly_express as px

st.set_page_config(page_title='Historical Weather Data')

st.markdown('# Historical Weather Data')

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

col1, col2 = st.columns([1,2])

with col1:
	st.header('De Historical Weather DataFrame bevat de volgende kolommen:')
	st.markdown(
	'''  
	- **Date:**  
	  *Bevat de datums van alle datapunten. Elk rij in dit kolom is een nieuwe dag.*  
	- **temperature_2m_mean:**  
	  *Bevat de gemiddelde temperatuur, in graden Celsius, dat op twee meter hoog is gemeten.*  
	- **apparent_temperature_mean:**  
	  *Bevat de gemiddelde gevoels temperatuur, in graden Celsius, dat op twee meter is gemeten.*  
	- **daylight_duration:**  
	  *Bevat hoelang, in uren, het op een dag daglicht is.*  
	- **sunshine_duration:**  
	  *Bevat hoelang, in uren, de zon op een dag schijnt.*  
	- **precipitation_sum:**  
	  *Bevat de som, in millimeter, van de hoeveelheid gevallen neerslag.*  
	- **rain_sum:**  
	  *Bevat de som, in millimeter, van de hoeveelheid gevallen regen.*  
	- **snowfall_sum:**  
	  *Bevat de som, in millimeter, van de hoeveelheid gevallen sneeuw.*  
	- **precipitation_hours:**  
	  *Bevat hoelang, in uren, het regent op een dag.*  
	- **wind_speed_10m_max:**  
	  *Bevat, in km/u, de maximale wind snelheid op een dag.*  
	- **wind_gusts_10m_max:**  
	  *Bevat, in km/u, de maximale windvlaag snelheid op een dag.*  
	- **wind_direction_10m_dominant:**  
	  *Bevat, in graden, de dominante wind directie.*  
	'''
	)


with col2:
	st.header('Historical Weather DataFrame')
	daily_dataframe

st.text("")

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    st.header("Plotting Variable 1")
    variable1 = st.selectbox("Select first variable:", [
	'date', 'temperature_2m_mean', 'apparent_temperature_mean',
	'daylight_duration', 'sunshine_duration', 'precipitation_sum',
	'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max',
	'wind_gusts_10m_max', 'wind_direction_10m_dominant'
	])
			     
			     
			     
			 #     [
    #     'temperature_2m_mean', 'apparent_temperature_mean',
    #     'daylight_duration', 'sunshine_duration', 'precipitation_sum'
    # ])

with col2:
    st.header("Plotting Variable 2")
    variable2 = st.selectbox("Select second variable:", [
	'date', 'temperature_2m_mean', 'apparent_temperature_mean',
	'daylight_duration', 'sunshine_duration', 'precipitation_sum',
	'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max',
	'wind_gusts_10m_max', 'wind_direction_10m_dominant'
	])
			     
			     
			 #     [
    #     'rain_sum', 'snowfall_sum', 'precipitation_hours',
    #     'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant'
    # ])

with col3:
    st.header("Plot")
    
    if variable1 and variable2:
        fig = px.scatter(daily_dataframe, x=variable1, y=variable2, title=f'{variable1} vs {variable2}')
        st.plotly_chart(fig)




# fig = go.Figure()

# for x in ['daylight_duration', 'precipitation_sum', 'precipitation_hours']:
#     fig.add_trace(go.Scatter(
#         x=daily_dataframe[x], y=daily_dataframe['temperature_2m_mean'],
#         mode='markers', name=x
#     ))

# dropdown_buttons = [
#     {'label': 'daylight_duration', 'method': 'update',
#      'args': [{'visible': [True, False, False]},
#               {'title': 'Daylight Duration'}]},
    
#     {'label': 'sunshine_duration', 'method': 'update',
#      'args': [{'visible': [False, True, False]},
#               {'title': 'Precipitation Sum'}]},    
    
#     {'label': 'precipitation_hours', 'method': 'update',
#      'args': [{'visible': [False, False, True]},
#               {'title': 'Precipitation Hours'}]},        
# ]

# fig.update_layout(
#     updatemenus=[{
#         'type': 'dropdown',
#         'x': 1.1, 'y': 1.15,
#         'showactive': True,
#         'active': 0,
#         'buttons': dropdown_buttons
#     }],
#     title={'text': 'Gemiddelde temperatuur in Amsterdam'},
#     xaxis={'title': {'text': 'Tijd [uur]'}},               
#     yaxis={'title': {'text': 'Temperatuur [°C]'}},         
# )

# fig.data[1].visible=False
# fig.data[2].visible=False

# st.plotly_chart(fig, use_container_width=True)
