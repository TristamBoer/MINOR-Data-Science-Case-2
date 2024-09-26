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
import seaborn as sns
import matplotlib.pyplot as plt

def page_config():
        st.set_page_config(layout='wide',
                          page_title="Climate Models")
page_config()

st.markdown('# Climate Change')
st.markdown(
	'''  
	Binnen de Climate Change API worden de volgende onderwerpen op volgorde besproken:    
        - **KNMI & OpenMeteo temperatuur voorspelling**      
        - **Temperatuur voorspelling in verschillende steden**      
        - **Temperatuur bij 'La Niña' en 'El Niño'**    
        - **Regen bij 'La Niña' en 'El Niño'**    
        - **Gevallen regen per jaar & maand**  
	'''
	)


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

st.header('KNMI & OpenMeteo temperatuur voorspelling')

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

st.header('Temperatuur voorspelling in verschillende steden')

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


col1, col2 = st.columns(2)

with col1:
        # Streamlit Radio Buttons for City Selection (Single Selection)
        selected_city = st.radio("Select City", options=cities, index=0)

with col2:
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



@st.cache_data
def data4():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # API Request parameters
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": 52.37403,
        "longitude": 4.88969,
        "start_date": "1950-01-01",
        "end_date": "2023-12-31",
        "models": "EC_Earth3P_HR",
        "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "rain_sum"]
    }
    
    responses = openmeteo.weather_api(url, params=params)
    
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    
    # Process daily data
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
    
    # Prepare DataFrame
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_mean": daily_temperature_2m_mean,
        "temperature_2m_max": daily_temperature_2m_max,
        "temperature_2m_min": daily_temperature_2m_min,
        "rain_sum": daily_rain_sum
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_data)
    
    # Ensure that the numeric columns are of type float (you can cast other data types if necessary)
    df["temperature_2m_mean"] = pd.to_numeric(df["temperature_2m_mean"], errors="coerce")
    df["temperature_2m_max"] = pd.to_numeric(df["temperature_2m_max"], errors="coerce")
    df["temperature_2m_min"] = pd.to_numeric(df["temperature_2m_min"], errors="coerce")
    df["rain_sum"] = pd.to_numeric(df["rain_sum"], errors="coerce")
    
    return df

# Fetch data
daily_dataframe = data4()

# Set 'date' as the index
daily_dataframe.set_index('date', inplace=True)

# Resample the data per year and calculate the mean values
# Handle missing data or invalid values during resampling by using the mean with `skipna=True`
yearly_dataframe = daily_dataframe.resample('Y').mean()

daily_dataframe = data4()

# Ensure 'date' column is properly parsed and set as the index
daily_dataframe.set_index('date', inplace=True)

# Convert columns to numeric, ensuring all values can be averaged (handling invalid data)
daily_dataframe["temperature_2m_mean"] = pd.to_numeric(daily_dataframe["temperature_2m_mean"], errors="coerce")
daily_dataframe["temperature_2m_max"] = pd.to_numeric(daily_dataframe["temperature_2m_max"], errors="coerce")
daily_dataframe["temperature_2m_min"] = pd.to_numeric(daily_dataframe["temperature_2m_min"], errors="coerce")
daily_dataframe["rain_sum"] = pd.to_numeric(daily_dataframe["rain_sum"], errors="coerce")

# Resample the data per year and calculate the mean values
yearly_dataframe = daily_dataframe.resample('Y').mean()

# Reset the index and create a 'year' column from the 'date'
yearly_dataframe.reset_index(inplace=True)
yearly_dataframe['year'] = yearly_dataframe['date'].dt.year  # Extra column for the year

# Resample the data per month and calculate the mean values
monthly_dataframe = daily_dataframe.resample('M').mean()

# Reset the index to work with month values easily, if needed
monthly_dataframe.reset_index(inplace=True)

# Voeg een kolom toe voor de maand
monthly_dataframe['month'] = monthly_dataframe['date'].dt.month

# Optioneel: Voeg een kolom toe voor de maandnaam
monthly_dataframe['month_name'] = monthly_dataframe['date'].dt.strftime('%B')
#%%
# Lijst met La Niña jaren als strings
la_nina_years = ['1954', '1955', '1964', '1970', '1971', '1973', '1974', '1975', '1983', '1984', 
                 '1988', '1995', '1998', '1999', '2000', '2005', '2007', '2008', '2010', 
                 '2011', '2016', '2017', '2020', '2021', '2022']


# Toevoegen van een nieuwe kolom met 'La Niña' of 'El Niño' op basis van de jaarcontrole in yearly_dataframe
yearly_dataframe['Oceanic Niño Index'] = yearly_dataframe['year'].astype(str).isin(la_nina_years).map({True: 'La Niña', False: 'El Niño'})

# Toevoegen van een nieuwe kolom met 'La Niña' of 'El Niño' op basis van de jaarcontrole in monthly_dataframe
monthly_dataframe['Oceanic Niño Index'] = monthly_dataframe['date'].dt.year.astype(str).isin(la_nina_years).map({True: 'La Niña', False: 'El Niño'})

colors = px.colors.qualitative.Set1

st.header("Temperatuur bij 'La Niña' en 'El Niño'")

fig = px.scatter(
    data_frame=yearly_dataframe, x='year', y='temperature_2m_mean',
    trendline='ols',
    marginal_x='histogram', marginal_y='histogram',
    color='Oceanic Niño Index', color_discrete_sequence=colors, height=800,
    title='Gemiddelde tempratuur per jaar met trendline'
    )

fig.update_layout(
    yaxis_title='Gemiddelde tempratuur in graden celsius voor Amsterdam',
    xaxis_title='Jaar',
    title_font_size=18,
    legend_title_text='Oceanic Niño Index'
    )

st.plotly_chart(fig)

st.header("Regen bij 'La Niña' en 'El Niño'")

plt.figure(figsize=(12,6))
sns.barplot(x='month_name', y='rain_sum', hue='Oceanic Niño Index', data=monthly_dataframe, palette='Set1')

# Titels en labels toevoegen
plt.title('Som van de regen per maand met Oceanic Niño Index')
plt.xlabel('Maand')
plt.ylabel('Som van de regen (mm)')

# Legenda toevoegen
plt.legend(title='Oceanic Niño Index')

# Grafiek weergeven
plt.tight_layout()
st.pyplot(plt)

st.markdown(
        '''
        - **[Oceanic Niño Index](https://ggweather.com/enso/oni.htm)**
        '''
)


@st.cache_data
def data5():
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        
        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
        	"latitude": 52.25,
        	"longitude": 5.75,
        	"start_date": "2000-01-02",
        	"end_date": "2024-09-19",
        	"hourly": ["precipitation", "rain"],
        	"daily": ["temperature_2m_mean", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours"],
        	"timezone": "auto"
        }
        responses = openmeteo.weather_api(url, params=params)
        
        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        
        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
        hourly_rain = hourly.Variables(1).ValuesAsNumpy()
        
        hourly_data = {"date": pd.date_range(
        	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        	freq = pd.Timedelta(seconds = hourly.Interval()),
        	inclusive = "left"
        )}
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["rain"] = hourly_rain
        
        hourly_dataframe = pd.DataFrame(data = hourly_data)
        
        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
        daily_snowfall_sum = daily.Variables(3).ValuesAsNumpy()
        daily_precipitation_hours = daily.Variables(4).ValuesAsNumpy()
        
        daily_data = {"date": pd.date_range(
        	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        	freq = pd.Timedelta(seconds = daily.Interval()),
        	inclusive = "left"
        )}
        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["precipitation_sum"] = daily_precipitation_sum
        daily_data["rain_sum"] = daily_rain_sum
        daily_data["snowfall_sum"] = daily_snowfall_sum
        daily_data["precipitation_hours"] = daily_precipitation_hours
        
        return pd.DataFrame(data = daily_data)
daily_dataframe = data5()

month_name = [
    "",  # Index 0 is unused; placeholders for 1-indexed months
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

data = daily_dataframe.dropna(subset=['date'])

daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'])
daily_dataframe['year'] = daily_dataframe['date'].dt.year
daily_dataframe['month'] = daily_dataframe['date'].dt.month

year_list = daily_dataframe['year'].unique()
start_year = int(year_list.min())
end_year = int(year_list.max())

st.header('Gevallen regen per jaar & maand')

# Create a slider for year selection, using start_year and end_year as the limits
year_selection = st.slider("Select a year", start_year, end_year, value=start_year)

# Create a dropdown for month selection, with options being "All Months" alongside named months
month_options = ['All Months'] + [month_name[i] for i in range(1, 13)]
month_selection = st.selectbox("Select a month", month_options)

# Filter based on the selected year
filtered_data = daily_dataframe[daily_dataframe['year'] == year_selection]


# Further filter based on selected month
if month_selection != 'All Months':
    month_selection_index = month_options.index(month_selection)
    filtered_data = filtered_data[filtered_data['month'] == month_selection_index]

# Melt the DataFrame to show full precipitation in both rain and snow
melted_data = pd.melt(filtered_data, id_vars='date', value_vars=['rain_sum', 'snowfall_sum'],
                       var_name='precipitation_type', value_name='amount')

# Create a bar plot based on the melted dataframe, showing total rain and snow on each day.
fig = px.bar(melted_data, 
              x='date', 
              y='amount', 
              color='precipitation_type', 
              title=f'Total rainfall for {year_selection} {month_selection if month_selection != "All Months" else ""}',
              labels={'amount': 'Sum of rainfall (mm)', 'date': 'Date'},
              color_discrete_sequence=px.colors.qualitative.Set1)

# Update x-axis ticks based on the month selection
if month_selection == 'All Months':
    fig.update_xaxes(
        dtick="M1",  # One tick per month
        tickformat="%Y-%m",  # Format as YYYY-MM
        tickangle=90  # Rotate x-axis labels
    )
else:
    fig.update_xaxes(
        dtick="D1",  # One tick per day
        tickformat="%Y-%m-%d",  # Format as YYYY-MM-DD
        tickangle=90  # Rotate x-axis labels
    )

# Show the plot
fig.update_layout(legend_title_text='Precipitation Type')
st.plotly_chart(fig)
