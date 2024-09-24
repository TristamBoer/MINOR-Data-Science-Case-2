import streamlit as st
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

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
	"start_date": "1950-01-01",
	"end_date": "2024-01-01",
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "is_day", "sunshine_duration"],
	"daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours"],
	"timezone": "Europe/Berlin"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_weather_code = daily.Variables(0).ValuesAsNumpy()
daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
daily_temperature_2m_mean = daily.Variables(3).ValuesAsNumpy()
daily_apparent_temperature_max = daily.Variables(4).ValuesAsNumpy()
daily_apparent_temperature_min = daily.Variables(5).ValuesAsNumpy()
daily_apparent_temperature_mean = daily.Variables(6).ValuesAsNumpy()
daily_sunrise = daily.Variables(7).ValuesAsNumpy()
daily_sunset = daily.Variables(8).ValuesAsNumpy()
daily_daylight_duration = daily.Variables(9).ValuesAsNumpy()
daily_sunshine_duration = daily.Variables(10).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(11).ValuesAsNumpy()
daily_rain_sum = daily.Variables(12).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(13).ValuesAsNumpy()
daily_precipitation_hours = daily.Variables(14).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
	start=pd.to_datetime(daily.Time(), unit="s", utc=True),
	end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
	freq=pd.Timedelta(seconds=daily.Interval()),
	inclusive="left"
	),
	"weather_code": daily_weather_code, "temperature_2m_max": daily_temperature_2m_max,
	"temperature_2m_min": daily_temperature_2m_min, "temperature_2m_mean": daily_temperature_2m_mean,
	"apparent_temperature_max": daily_apparent_temperature_max,
	"apparent_temperature_min": daily_apparent_temperature_min,
	"apparent_temperature_mean": daily_apparent_temperature_mean,
	"daylight_duration": daily_daylight_duration, "sunshine_duration": daily_sunshine_duration,
	"precipitation_sum": daily_precipitation_sum, "rain_sum": daily_rain_sum, "snowfall_sum": daily_snowfall_sum,
	"precipitation_hours": daily_precipitation_hours}

daily_dataframe = pd.DataFrame(data=daily_data)
daily_dataframe['date'] = daily_dataframe['date'].dt.date
daily_dataframe.index = daily_dataframe['date']
daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'], yearfirst=True)














st.set_page_config(page_title='Historical Weather Data')

st.markdown('# Historical Weather Data')

with st.expander('Data'):
  st.write('**Raw Data**')
  daily_dataframe
