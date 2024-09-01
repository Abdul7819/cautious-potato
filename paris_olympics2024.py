import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Suppress warnings for a cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

st.title("Paris Olympics 2024 Medal Tally Data Analysis")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file for Paris Olympics 2024 Medal Tally", type=["csv"])

if uploaded_file is not None:
    # Step 2: Read CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Step 3: Display basic information about the dataset
    st.write("### Dataset Preview")
    st.write(df.head())
    st.write("### Dataset Information")
    buffer = df.info(buf=None)
    st.text(buffer)
    
    # Step 4: Data Cleaning and Preparation
    st.write("### Checking for Missing Values")
    st.write(df.isnull().sum())

    # Summary statistics
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Verify 'Total Medals' calculation
    df['Total Medals Check'] = df['Gold'] + df['Silver'] + df['Bronze']
    df['Discrepancy'] = df['Total Medals Check'] - df['Total Medals']
    discrepancies = df[df['Discrepancy'] != 0]
    st.write("### Discrepancies in Total Medals Calculation")
    if discrepancies.empty:
        st.write("No discrepancies found in the total medals calculation.")
    else:
        st.write(discrepancies)
    
    # Step 5: Visualization
    st.write("## Visualizations")

    # Top 5 countries stacked bar chart
    top_5_countries = df.head(5)
    st.write("### Medal Distribution of Top 5 Countries")
    fig, ax = plt.subplots(figsize=(10, 6))
    top_5_countries.set_index('Country')[['Gold', 'Silver', 'Bronze']].plot(kind='bar', stacked=True, color=['gold', 'silver', '#cd7f32'], ax=ax)
    plt.title('Medal Distribution of Top 5 Countries in Paris 2024 Olympics')
    plt.xlabel('Country')
    plt.ylabel('Number of Medals')
    st.pyplot(fig)

    # Top 10 countries bar chart
    st.write("### Top 10 Countries by Total Medals")
    top_countries = df.sort_values('Total Medals', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Total Medals', y='Country', data=top_countries, palette='viridis', ax=ax)
    plt.title('Top 10 Countries by Total Medals in Paris 2024 Olympics')
    plt.xlabel('Total Medals')
    plt.ylabel('Country')
    st.pyplot(fig)

    # Distribution histograms for Gold, Silver, and Bronze medals
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

    # Step 6: Summary and Insights
    st.write("## Summary and Insights")
    st.write("""
    Based on the analysis of the Paris 2024 Olympics medal tally data, the United States led with the highest total medals, followed closely by China. 
    The distribution of gold, silver, and bronze medals shows that there is a competitive balance among the top-performing countries, 
    with many nations excelling in specific types of events. Further analysis can delve into specific sports or events to understand the strengths 
    and strategies of different countries in achieving their medal counts.
    """)
else:
    st.write("Please upload a CSV file to start analyzing the data.")
