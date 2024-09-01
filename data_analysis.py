import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Data Analysis Web App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("### Dataset Overview")
        st.write(df.head())

        # Show basic statistics
        st.write("### Basic Statistics")
        st.write(df.describe())

        # Analysis
        if st.checkbox('Show detailed analysis'):
            col = st.selectbox('Select column for analysis:', df.columns)

            if col:
                data = df[col].dropna()
                
                # Mean
                mean_val = np.mean(data)
                st.write(f"**Mean**: {mean_val}")

                # Median
                median_val = np.median(data)
                st.write(f"**Median**: {median_val}")

                # Mode
                mode_val = data.mode()[0]
                st.write(f"**Mode**: {mode_val}")

                # Standard Deviation
                std_dev = np.std(data)
                st.write(f"**Standard Deviation**: {std_dev}")

                # Variance
                variance = np.var(data)
                st.write(f"**Variance**: {variance}")

                # Percentiles
                percentiles = np.percentile(data, [25, 50, 75])
                st.write(f"**25th Percentile**: {percentiles[0]}")
                st.write(f"**50th Percentile (Median)**: {percentiles[1]}")
                st.write(f"**75th Percentile**: {percentiles[2]}")

                # Visualizations
                st.write("### Visualizations")

                # Histogram
                st.write("#### Histogram")
                fig, ax = plt.subplots()
                ax.hist(data, bins=30, edgecolor='black')
                ax.set_title('Histogram')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

                # Box Plot
                st.write("#### Box Plot")
                fig, ax = plt.subplots()
                sns.boxplot(x=data, ax=ax)
                ax.set_title('Box Plot')
                st.pyplot(fig)

                # Distribution Plot
                st.write("#### Distribution Plot")
                fig, ax = plt.subplots()
                sns.histplot(data, kde=True, ax=ax)
                ax.set_title('Distribution Plot')
                st.pyplot(fig)

                # Q-Q Plot
                st.write("#### Q-Q Plot")
                fig, ax = plt.subplots()
                sm.qqplot(data, line ='45', ax=ax)
                ax.set_title('Q-Q Plot')
                st.pyplot(fig)

if __name__ == "__main__":
    main()
