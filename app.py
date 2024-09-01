import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import requests
import folium
from streamlit_folium import folium_static
from datetime import datetime
import rasterio
from io import BytesIO
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
import io

# Function to display basic info about the dataframe
def display_dataframe_info(df):
    st.write("DataFrame Information:")
    st.write(df.info())
    st.write("First 5 rows:")
    st.write(df.head())

# Function to scale data
def scale_data(df, method):
    scaler = None
    if method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    if scaler:
        scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
        scaled_df = pd.DataFrame(scaled_data, columns=df.select_dtypes(include=[np.number]).columns)
        return pd.concat([scaled_df, df.select_dtypes(exclude=[np.number])], axis=1)
    return df

# Function to create dummy variables
def create_dummies(df, column):
    return pd.get_dummies(df, columns=[column])

# Function to perform data mapping
def map_data(df, column, mapping_dict):
    df[column] = df[column].map(mapping_dict)
    return df

# Function to display statistics
def display_statistics(df):
    st.write("Dataset Statistics:")
    st.write(df.describe())

# Function to create different types of plots
def create_plot(df, plot_type, x_column, y_column=None):
    plt.figure(figsize=(10, 6))
    if plot_type == 'Scatter Plot':
        sns.scatterplot(data=df, x=x_column, y=y_column)
    elif plot_type == 'Line Graph':
        sns.lineplot(data=df, x=x_column, y=y_column)
    elif plot_type == 'Box Plot':
        sns.boxplot(data=df, x=x_column)
    elif plot_type == 'Histogram':
        sns.histplot(df[x_column], kde=True)
    elif plot_type == 'Pie Chart':
        df[x_column].value_counts().plot.pie(autopct='%1.1f%%')
    st.pyplot(plt)

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

# Function to correct missing scan lines using nearest neighbor interpolation
def correct_missing_scanlines(image):
    corrected_image = np.copy(image)
    for band in range(image.shape[2]):
        band_data = image[:, :, band]
        zero_indices = np.argwhere(band_data == 0)
        for row, col in zero_indices:
            neighbors = []
            if row > 0 and band_data[row - 1, col] != 0:
                neighbors.append(band_data[row - 1, col])
            if row < band_data.shape[0] - 1 and band_data[row + 1, col] != 0:
                neighbors.append(band_data[row + 1, col])
            if col > 0 and band_data[row, col - 1] != 0:
                neighbors.append(band_data[row, col - 1])
            if col < band_data.shape[1] - 1 and band_data[row, col + 1] != 0:
                neighbors.append(band_data[row, col + 1])
            if neighbors:
                corrected_image[row, col, band] = np.mean(neighbors)
    return corrected_image

# Load dataset from uploaded file
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Function to preprocess the input data
def preprocess_input(example_data, encoders):
    for col in example_data.columns:
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels by mapping to an integer (e.g. -1 for unknown)
            example_data[col] = [le.transform([value])[0] if value in le.classes_ else -1 for value in example_data[col]]
    return example_data

# Function to get latitude and longitude for a given location
def get_location_coordinates(location):
    location_coordinates = {
        "Dubai": [25.276987, 55.296249],
        "Singapore": [1.352083, 103.819839],
        "Berlin": [52.520008, 13.404954],
        "Tokyo": [35.682839, 139.759455],
        "San Francisco": [37.774929, -122.419418],
        "London": [51.507351, -0.127758],
        "Paris": [48.856613, 2.352222],
        "Sydney": [-33.868820, 151.209296],
        "Toronto": [43.651070, -79.347015],
        "New York": [40.712776, -74.005974]
    }
    return location_coordinates.get(location, [40.712776, -74.005974])  # Default to New York if location not found

# Streamlit app
st.title("Unified Data Processing and Visualization App")

# Upload CSV file
uploaded_file_data = st.file_uploader("Upload your CSV file for Data Processing", type=["csv"], key="data_processing")

if uploaded_file_data is not None:
    df = load_data(uploaded_file_data)
    display_dataframe_info(df)

    # Scaling options
    st.sidebar.subheader("Scaling Options")
    scaling_method = st.sidebar.selectbox("Select Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
    if scaling_method != "None":
        df = scale_data(df, scaling_method)
        st.write(f"Data after {scaling_method}:")
        st.write(df.head())

    # Dummy variable creation
    st.sidebar.subheader("Dummy Variables")
    dummy_column = st.sidebar.selectbox("Select column for dummies", [None] + list(df.select_dtypes(include=[object]).columns))
    if dummy_column:
        df = create_dummies(df, dummy_column)
        st.write(f"Data after creating dummies for {dummy_column}:")
        st.write(df.head())

    # Data mapping
    st.sidebar.subheader("Data Mapping")
    map_column = st.sidebar.selectbox("Select column for mapping", [None] + list(df.columns))
    if map_column:
        mapping_input = st.sidebar.text_area("Enter mapping dictionary (e.g. {'A': 1, 'B': 2})")
        if mapping_input:
            mapping_dict = eval(mapping_input)
            df = map_data(df, map_column, mapping_dict)
            st.write(f"Data after mapping {map_column}:")
            st.write(df.head())

    # Display statistics
    display_statistics(df)

    # Plot options
    st.sidebar.subheader("Plot Options")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["None", "Scatter Plot", "Line Graph", "Box Plot", "Histogram", "Pie Chart"])
    if plot_type != "None":
        x_column = st.sidebar.selectbox("Select X column", df.columns)
        y_column = None
        if plot_type in ["Scatter Plot", "Line Graph"]:
            y_column = st.sidebar.selectbox("Select Y column", df.columns)
        create_plot(df, plot_type, x_column, y_column)

# Weather Prediction
st.title("7-Day Weather Forecast for Pakistan")
cities = ['Karachi', 'Lahore', 'Islamabad', 'Faisalabad', 'Rawalpindi', 'Multan']
selected_city = st.selectbox("Select a city", cities)
api_key = 'f28433171728fce993d9c3ac5b3db522'

if st.button('Get Weather'):
    weather_data = get_weather(selected_city, api_key)
    if weather_data['cod'] == '200':
        st.write(f"Weather Forecast for {selected_city}")
        forecast_times = [datetime.fromtimestamp(item['dt']) for item in weather_data['list']]
        forecast_dates = sorted(set([dt.date() for dt in forecast_times]))
        selected_date = st.selectbox("Select a date", forecast_dates)

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

                st.write(f"Date and Time: {date}")
                st.write(f"Temperature: {temp} °C")
                st.write(f"Weather: {weather.capitalize()}")
                st.write(f"Humidity: {humidity}%")
                st.write(f"Pressure: {pressure} hPa")
                st.write(f"Wind Speed: {wind_speed} m/s")
                st.write("---")

            lat = weather_data['city']['coord']['lat']
            lon = weather_data['city']['coord']['lon']
            map_ = create_map(lat, lon)
            folium_static(map_)
        else:
            st.write(f"No weather data available for {selected_date}.")
    else:
        st.write(f"Error: {weather_data['message']}")

# Student Health Clustering
st.title("Student Mental Health Survey Clustering")
uploaded_file_health = st.file_uploader("Choose a CSV file for Health Data", type="csv", key="health_clustering")

if uploaded_file_health is not None:
    df = load_data(uploaded_file_health)
    st.write("### Dataset Overview")
    st.write(df.head())
    st.write(df.describe())
    st.write(f"Data shape: {df.shape}")

    # Feature selection
    st.sidebar.header("Select Features")
    num_features = [col for col in df.columns if df[col].dtype != 'object']
    cat_features = [col for col in df.columns if df[col].dtype == 'object']

    selected_num_features = st.sidebar.multiselect("Select numerical features", num_features, default=num_features)
    selected_cat_features = st.sidebar.multiselect("Select categorical features", cat_features, default=cat_features)

    if selected_num_features or selected_cat_features:
        # Encoding categorical features
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in selected_cat_features:
            df_encoded[col] = le.fit_transform(df_encoded[col])

        # Scaling features
        scaler = MinMaxScaler()
        df_scaled = df_encoded.copy()
        df_scaled[selected_num_features] = scaler.fit_transform(df_encoded[selected_num_features])

        data = df_scaled[selected_num_features + selected_cat_features].values

        # Split data
        x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)

        # KMeans clustering
        num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=2)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)

        st.write(f"### Silhouette Score: {score:.2f}")

        # Display clustering results
        st.write("### Clustering Results:")
        df_scaled['Cluster'] = labels
        st.write(df_scaled.head())

        # Plot clusters
        if len(selected_num_features) >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(df_scaled[selected_num_features[0]], df_scaled[selected_num_features[1]], c=df_scaled['Cluster'], cmap='viridis')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            ax.set_xlabel(selected_num_features[0])
            ax.set_ylabel(selected_num_features[1])
            st.pyplot(fig)

        # New data input form
        st.sidebar.header("Classify New Data")
        new_data = {}

        for feature in selected_num_features:
            value = st.sidebar.number_input(f"Enter value for {feature}", value=0.0)
            new_data[feature] = value

        for feature in selected_cat_features:
            options = df[feature].unique()
            selected_option = st.sidebar.selectbox(f"Select value for {feature}", options)
            new_data[feature] = selected_option

        if st.sidebar.button("Classify New Data"):
            # Prepare new data for prediction
            new_df = pd.DataFrame([new_data])
            new_df_encoded = new_df.copy()

            # Encoding new data
            for col in selected_cat_features:
                new_df_encoded[col] = le.transform(new_df_encoded[col])

            # Scaling new data
            new_df_scaled = new_df_encoded.copy()
            new_df_scaled[selected_num_features] = scaler.transform(new_df_encoded[selected_num_features])

            new_data_scaled = new_df_scaled[selected_num_features + selected_cat_features].values
            cluster = kmeans.predict(new_data_scaled)

            st.write(f"### New Data Classification")
            st.write(f"The new data point belongs to cluster: {cluster[0]}")

# US Airlines Data
st.title("Flight Data Analysis App")

uploaded_file_flight = st.file_uploader("Choose a CSV file for Flight Data", type="csv", key="flight_data")

if uploaded_file_flight is not None:
    df = load_data(uploaded_file_flight)

    # Display basic information about the dataset
    st.write("### Dataset Overview")
    st.write("#### First few rows:")
    st.write(df.head())

    st.write("#### Last few rows:")
    st.write(df.tail())

    st.write(f"#### Shape of the dataset: {df.shape}")
    st.write("#### Data Info:")
    st.write(df.info())

    st.write("#### Statistical Summary:")
    st.write(df.describe().T)

    # Plot a bar chart for a statistical summary
    st.write("### Statistical Summary Bar Chart")
    fig, ax = plt.subplots(figsize=(14, 7))
    df.describe(include='all').plot(kind='bar', ax=ax)
    ax.set_title('Statistical Summary of Dataset')
    st.pyplot(fig)

    # Check for missing values
    st.write("### Missing Values")
    st.write("#### Number of missing values per column:")
    st.write(df.isna().sum())

    # Visualize missing values using a heatmap
    st.write("#### Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values Heatmap')
    st.pyplot(fig)

    # Fill missing values with 0 (or another strategy)
    df = df.fillna(0)

    # Check for duplicated rows
    st.write(f"#### Duplicated rows: {df.duplicated().sum()}")

    # Correlation matrix and heatmap
    numeric_df = df.select_dtypes(include=[float, int])
    correlation_matrix = numeric_df.corr()

    st.write("### Correlation Matrix Heatmap")
    fig = sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r')
    st.pyplot(fig)

    # Columns to plot
    columns = ['Company', 'Company Score', 'Job Title', 'Location', 'Salary']

    st.write("### Column Distributions")
    for column in columns:
        if column in df.columns:
            try:
                if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                    column_counts = df[column].value_counts().reset_index()
                    column_counts.columns = [column, 'count']

                    fig = px.bar(column_counts, x=column, y='count', title=f'Distribution of {column}',
                                 labels={column: column, 'count': 'Count'}, text='count')
                    fig.update_layout(xaxis_title=column, yaxis_title='Count',
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      title_font=dict(size=18, family="Arial"), xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig)

                elif df[column].dtype in ['int64', 'float64']:
                    fig = px.histogram(df, x=column, title=f'Distribution of {column}',
                                       labels={column: column, 'count': 'Count'})
                    fig.update_layout(xaxis_title=column, yaxis_title='Count',
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      title_font=dict(size=18, family="Arial"))
                    st.plotly_chart(fig)
            except Exception as e:
                st.write(f"Could not create plot for column {column}: {e}")

# Paris Olympics 2024 Medal Tally
st.title("Paris Olympics 2024 Medal Tally Data Analysis")
uploaded_file_olympics = st.file_uploader("Upload your CSV file for Paris Olympics 2024 Medal Tally", type=["csv"], key="olympics_data")

if uploaded_file_olympics is not None:
    df = load_data(uploaded_file_olympics)

    st.write("### Dataset Preview")
    st.write(df.head())
    st.write("### Dataset Information")
    buffer = df.info(buf=None)
    st.text(buffer)

    st.write("### Checking for Missing Values")
    st.write(df.isnull().sum())

    st.write("### Summary Statistics")
    st.write(df.describe())

    df['Total Medals Check'] = df['Gold'] + df['Silver'] + df['Bronze']
    df['Discrepancy'] = df['Total Medals Check'] - df['Total Medals']
    discrepancies = df[df['Discrepancy'] != 0]
    st.write("### Discrepancies in Total Medals Calculation")
    if discrepancies.empty:
        st.write("No discrepancies found in the total medals calculation.")
    else:
        st.write(discrepancies)

    st.write("## Visualizations")

    top_5_countries = df.head(5)
    st.write("### Medal Distribution of Top 5 Countries")
    fig, ax = plt.subplots(figsize=(10, 6))
    top_5_countries.set_index('Country')[['Gold', 'Silver', 'Bronze']].plot(kind='bar', stacked=True, color=['gold', 'silver', '#cd7f32'], ax=ax)
    plt.title('Medal Distribution of Top 5 Countries in Paris 2024 Olympics')
    plt.xlabel('Country')
    plt.ylabel('Number of Medals')
    st.pyplot(fig)

    st.write("### Top 10 Countries by Total Medals")
    top_countries = df.sort_values('Total Medals', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Total Medals', y='Country', data=top_countries, palette='viridis', ax=ax)
    plt.title('Top 10 Countries by Total Medals in Paris 2024 Olympics')
    plt.xlabel('Total Medals')
    plt.ylabel('Country')
    st.pyplot(fig)

    st.write("### Distribution of Medals")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df['Gold'], bins=15, kde=True, color='gold', ax=axs[0])
    axs[0].set_title('Distribution of Gold Medals')

    sns.histplot(df['Silver'], bins=15, kde=True, color='silver', ax=axs[1])
    axs[1].set_title('Distribution of Silver Medals')

    sns.histplot(df['Bronze'], bins=15, kde=True, color='#cd7f32', ax=axs[2])
    axs[2].set_title('Distribution of Bronze Medals')

    plt.tight_layout()
    st.pyplot(fig)

    st.write("## Summary and Insights")
    st.write("""
    Based on the analysis of the Paris 2024 Olympics medal tally data, the United States led with the highest total medals, followed closely by China. 
    The distribution of gold, silver, and bronze medals shows that there is a competitive balance among the top-performing countries, 
    with many nations excelling in specific types of events. Further analysis can delve into specific sports or events to understand the strengths 
    and strategies of different countries in achieving their medal counts.
    """)

# Sentinel-2 Image Analysis
st.title("Sentinel-2 Image Analysis: NDVI, Water Bodies, and Buildup Landcover")
uploaded_file_sentinel = st.file_uploader("Upload Sentinel-2 Image", type=["tif"], key="sentinel_data")

if uploaded_file_sentinel is not None:
    with rasterio.open(BytesIO(uploaded_file_sentinel.read())) as src:
        band_red = src.read(3)  # Assuming band 4 is red
        band_nir = src.read(4)  # Assuming band 8 is NIR

    # Calculate NDVI
    ndvi = (band_nir - band_red) / (band_nir + band_red + 1e-10)

    # Classify landcover
    water_landcover = np.where(ndvi < 0.1, 1, 0)
    buildup_landcover = np.where(ndvi > 0.3, 1, 0)

    st.write("**NDVI**")
    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.title("NDVI")
    st.pyplot(plt)

    st.write("**Water Bodies (NDVI < 0.1)**")
    plt.figure(figsize=(10, 10))
    plt.imshow(water_landcover, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Water Landcover")
    st.pyplot(plt)

    st.write("**Buildup Landcover (NDVI > 0.3)**")
    plt.figure(figsize=(10, 10))
    plt.imshow(buildup_landcover, cmap='Oranges', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Buildup Landcover")
    st.pyplot(plt)

# Salary Prediction using Random Forest
st.title("Salary Prediction Web App")

job_title_options = ['Data Scientist', 'Software Engineer', 'Product Manager']
industry_options = ['Technology', 'Finance', 'Healthcare']
company_size_options = ['Small', 'Medium', 'Large']
location_options = ['San Francisco', 'New York', 'London']
ai_adoption_level_options = ['Low', 'Medium', 'High']
automation_risk_options = ['Low', 'Medium', 'High']
required_skills_options = ['Python', 'SQL', 'Machine Learning']
remote_friendly_options = ['Yes', 'No']
job_growth_projection_options = ['Positive', 'Negative']

job_title = st.selectbox("Job Title", job_title_options)
industry = st.selectbox("Industry", industry_options)
company_size = st.selectbox("Company Size", company_size_options)
location = st.selectbox("Location", location_options)
ai_adoption_level = st.selectbox("AI Adoption Level", ai_adoption_level_options)
automation_risk = st.selectbox("Automation Risk", automation_risk_options)
required_skills = st.selectbox("Required Skills", required_skills_options)
remote_friendly = st.selectbox("Remote Friendly", remote_friendly_options)
job_growth_projection = st.selectbox("Job Growth Projection", job_growth_projection_options)

input_data = pd.DataFrame({
    "Job_Title": [job_title],
    "Industry": [industry],
    "Company_Size": [company_size],
    "Location": [location],
    "AI_Adoption_Level": [ai_adoption_level],
    "Automation_Risk": [automation_risk],
    "Required_Skills": [required_skills],
    "Remote_Friendly": [remote_friendly],
    "Job_Growth_Projection": [job_growth_projection]
})

# Normally, you would load a pre-trained model
# For demonstration, we are using a RandomForestClassifier without training

model = RandomForestClassifier()  # Placeholder model, no actual training
# model.predict should be used with a trained model.

if st.button('Predict Salary'):
    st.write("Predicted Salary: $100,000")  # Placeholder for the actual prediction

    latitude, longitude = get_location_coordinates(location)

    # Create a base map
    m = folium.Map(location=[latitude, longitude], zoom_start=12)

    folium.Marker([latitude, longitude],
                  popup=f"Predicted Salary: $100,000",
                  icon=folium.Icon(color="blue")).add_to(m)

    folium_static(m)

# NASA Data Classification
st.title('NEO Classification App')

model_option = st.sidebar.selectbox(
    'Choose a Classification Model',
    ['Random Forest', 'XGBoost', 'LightGBM']
)

X = pd.DataFrame({
    'absolute_magnitude': np.random.rand(100),
    'estimated_diameter_min': np.random.rand(100),
    'estimated_diameter_max': np.random.rand(100),
    'relative_velocity': np.random.rand(100)
})
y = np.random.choice([0, 1], size=(100,))

somte = SMOTE(random_state=42)
X, y = somte.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)

clf = None
if model_option == 'Random Forest':
    clf = RandomForestClassifier(random_state=42)
elif model_option == 'XGBoost':
    clf = XGBClassifier(random_state=42)
else:
    clf = LGBMClassifier(random_state=42)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

st.subheader('Model Evaluation')
st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix Heatmap')
st.pyplot(fig)

# Learning Curve
st.subheader('Learning Curve')
train_sizes, train_scores, test_scores = learning_curve(
    clf, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.grid()
ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
ax.set_title("Learning Curves")
ax.set_xlabel("Training examples")
ax.set_ylabel("Score")
ax.legend(loc="best")
st.pyplot(fig)

