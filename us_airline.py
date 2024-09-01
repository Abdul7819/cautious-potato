import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import io

# Initialize the Streamlit app
def main():
    st.title("Flight Data Analysis App")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

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
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                        title='Correlation Matrix', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        
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
        
        # Generate and display a word cloud for the 'city1' column
        if 'city1' in df.columns:
            stop_words_list = set(STOPWORDS)
            counts = Counter(df["city1"].dropna().apply(lambda x: str(x)))
            
            wcc = WordCloud(background_color="black", width=1600, height=800, max_words=2000, stopwords=stop_words_list)
            wcc.generate_from_frequencies(counts)
            
            st.write("### Word Cloud for 'city1' Column")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wcc, interpolation='bilinear')
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig)
    else:
        st.write("Upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()
