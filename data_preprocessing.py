import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Function to display basic info about the dataframe
def display_dataframe_info(df):
    st.write("DataFrame Information:")
    st.write(df.info())
    st.write("First 5 rows:")
    st.write(df.head())

# Function to scale data
def scale_data(df, method):
    if method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    scaled_df = pd.DataFrame(scaled_data, columns=df.select_dtypes(include=[np.number]).columns)
    return pd.concat([scaled_df, df.select_dtypes(exclude=[np.number])], axis=1)

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
        sns.histplot(data=df[x_column], kde=True)
    elif plot_type == 'Pie Chart':
        df[x_column].value_counts().plot.pie(autopct='%1.1f%%')
    st.pyplot(plt)

# Streamlit app starts here
st.title("Data Cleaning Web App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
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
        mapping_input = st.sidebar.text_area("Enter mapping dictionary (e.g., {'A': 1, 'B': 2})")
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
