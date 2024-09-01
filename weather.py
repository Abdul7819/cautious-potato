import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
from datetime import datetime

# Function to get weather data
def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric',  # or 'imperial' for Fahrenheit
        'cnt': '40'  # Number of data points (3-hour intervals) in the forecast
    }
    response = requests.get(base_url, params=params)
    return response.json()

# Function to create a map with basemap
def create_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles='OpenStreetMap')  # Basemap tile layer
    folium.Marker(location=[lat, lon], popup="Location", icon=folium.Icon(color="blue")).add_to(m)
    return m

# Set up Streamlit app
st.title("7-Day Weather Forecast for Pakistan")

# Define cities in Pakistan
cities = ['Karachi', 'Lahore', 'Islamabad', 'Faisalabad', 'Rawalpindi', 'Multan']

# Add a selectbox for cities
selected_city = st.selectbox("Select a city", cities)

# Add your OpenWeatherMap API key here
api_key = 'f28433171728fce993d9c3ac5b3db522'

# Define Font Awesome icons
icons = {
    'Temperature': 'fa-thermometer-half',
    'Weather': 'fa-cloud-sun',
    'Humidity': 'fa-tint',
    'Pressure': 'fa-tachometer-alt',
    'Wind Speed': 'fa-wind'
}

# Font Awesome CSS link
font_awesome_css = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">'

if st.button('Get Weather'):
    weather_data = get_weather(selected_city, api_key)
    if weather_data['cod'] == '200':
        st.markdown(font_awesome_css, unsafe_allow_html=True)
        
        # Get coordinates for map
        lat = weather_data['city']['coord']['lat']
        lon = weather_data['city']['coord']['lon']
        
        # Extract forecast times and dates
        forecast_times = [datetime.fromtimestamp(item['dt']) for item in weather_data['list']]
        forecast_dates = sorted(set([dt.date() for dt in forecast_times]))

        # Add a date picker for selecting the forecast date
        selected_date = st.selectbox("Select a date", forecast_dates)

        # Filter data for the selected date
        daily_data = [item for item in weather_data['list'] if datetime.fromtimestamp(item['dt']).date() == selected_date]

        if daily_data:
            st.subheader(f"Weather Forecast for {selected_city} on {selected_date}")

            for day in daily_data:
                date = datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d %H:%M:%S')
                temp = day['main']['temp']
                weather = day['weather'][0]['description']
                humidity = day['main']['humidity']
                pressure = day['main']['pressure']
                wind_speed = day['wind']['speed']

                st.markdown(f"**Date and Time:** {date}")
                st.markdown(f"**Temperature:** <i class='fas {icons['Temperature']}' style='color:#FF4500;'></i> {temp} °C", unsafe_allow_html=True)
                st.markdown(f"**Weather:** <i class='fas {icons['Weather']}' style='color:#FFD700;'></i> {weather.capitalize()}", unsafe_allow_html=True)
                st.markdown(f"**Humidity:** <i class='fas {icons['Humidity']}' style='color:#00BFFF;'></i> {humidity}%", unsafe_allow_html=True)
                st.markdown(f"**Pressure:** <i class='fas {icons['Pressure']}' style='color:#32CD32;'></i> {pressure} hPa", unsafe_allow_html=True)
                st.markdown(f"**Wind Speed:** <i class='fas {icons['Wind Speed']}' style='color:#1E90FF;'></i> {wind_speed} m/s", unsafe_allow_html=True)
                st.write("---")

            st.subheader("Location on Map")
            map_ = create_map(lat, lon)
            folium_static(map_)
        else:
            st.write(f"No weather data available for {selected_date}.")
    else:
        st.write(f"Error: {weather_data['message']}")
