import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    st.title("AI-Based Job Market Analysis")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display basic info and dataset overview
        st.write("### Dataset Overview")
        st.write(df.head())

        st.write("### Basic Information")
        st.write("#### Column Names")
        st.write(df.columns)
        st.write("#### Missing Values")
        st.write(df.isna().sum())
        st.write("#### Duplicate Rows")
        st.write(df.duplicated().sum())

        st.sidebar.title("Select Analysis")
        analysis_options = [
            "Job Titles Distribution",
            "Industry Distribution",
            "Company Size Distribution",
            "Job Title by Company Size",
            "Industry by Company Size",
            "Location Analysis",
            "AI Adoption Level Distribution",
            "AI Adoption Level by Industry",
            "AI Adoption Level by Job Title",
            "Job Title by AI Adoption Level",
            "Automation Risk Distribution",
            "Automation Risk Analysis",
            "Required Skills Analysis",
            "Salary Distribution",
            "Salary by Industry and AI Adoption Level",
            "Salary by Location and Automation Risk",
            "Salary by Job Title and Automation Risk",
            "Remote Work Analysis",
            "Job Growth Projection Analysis"
        ]

        analysis_option = st.sidebar.selectbox("Select Analysis Type", analysis_options)

        if analysis_option == "Job Titles Distribution":
            st.write("### Job Titles Distribution")
            job_title_counts = df['Job_Title'].value_counts().sort_values(ascending=False)
            st.write(job_title_counts)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y='Job_Title', data=df, order=job_title_counts.index, ax=ax)
            ax.set_title('Count of Job Titles')
            st.pyplot(fig)

        elif analysis_option == "Industry Distribution":
            st.write("### Industry Distribution")
            industry_counts = df['Industry'].value_counts().sort_values(ascending=False)
            st.write(industry_counts)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y='Industry', data=df, order=industry_counts.index, ax=ax)
            ax.set_title('Count of Industry')
            st.pyplot(fig)

        elif analysis_option == "Company Size Distribution":
            st.write("### Company Size Distribution")
            fig, ax = plt.subplots(figsize=(5, 5))
            df["Company_Size"].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=20, ax=ax)
            ax.set_title('Distribution of Company Size')
            st.pyplot(fig)

        elif analysis_option == "Job Title by Company Size":
            st.write("### Job Title by Company Size")
            job_title_by_company_size = df.groupby("Company_Size")["Job_Title"].value_counts().unstack()
            fig, ax = plt.subplots(figsize=(12, 5))
            job_title_by_company_size.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Job Titles by Company Size')
            ax.set_xlabel('Job Title')
            ax.set_ylabel('Count')
            ax.legend(title='Company Size', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "Industry by Company Size":
            st.write("### Industry by Company Size")
            industry_by_company_size = df.groupby("Industry")["Company_Size"].value_counts().unstack()
            fig, ax = plt.subplots(figsize=(12, 5))
            industry_by_company_size.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Industry by Company Size')
            ax.set_xlabel('Industry')
            ax.set_ylabel('Count')
            ax.legend(title='Company Size', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "Location Analysis":
            st.write("### Location Analysis")
            st.write("#### Company Size by Location")
            st.write(df.groupby("Company_Size")["Location"].value_counts().unstack())
            st.write("#### Job Title by Location")
            st.write(df.groupby("Job_Title")["Location"].value_counts().unstack())
            st.write("#### Industry by Location")
            st.write(df.groupby("Industry")["Location"].value_counts().unstack())

        elif analysis_option == "AI Adoption Level Distribution":
            st.write("### AI Adoption Level Distribution")
            fig, ax = plt.subplots(figsize=(5, 5))
            df["AI_Adoption_Level"].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=20, ax=ax)
            ax.set_title('Distribution of AI Adoption Level')
            st.pyplot(fig)

        elif analysis_option == "AI Adoption Level by Industry":
            st.write("### AI Adoption Level by Industry")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=df, x='AI_Adoption_Level', hue='Industry', ax=ax)
            ax.set_title('AI Adoption Level Across Different Industries')
            ax.set_xlabel('AI Adoption Level')
            ax.set_ylabel('Count')
            ax.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "AI Adoption Level by Job Title":
            st.write("### AI Adoption Level by Job Title")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=df, x='AI_Adoption_Level', hue='Job_Title', ax=ax)
            ax.set_title('AI Adoption Level Across Different Job Titles')
            ax.set_xlabel('AI Adoption Level')
            ax.set_ylabel('Count')
            ax.legend(title='Job Title', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "Job Title by AI Adoption Level":
            st.write("### Job Title by AI Adoption Level")
            job_title_by_ai_adoption = df.groupby(["Job_Title", "AI_Adoption_Level"]).size().unstack()
            fig, ax = plt.subplots(figsize=(12, 5))
            job_title_by_ai_adoption.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Job Titles by AI Adoption Level')
            ax.set_xlabel('Job Title')
            ax.set_ylabel('Count')
            ax.legend(title='AI Adoption Level', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "Automation Risk Distribution":
            st.write("### Automation Risk Distribution")
            fig, ax = plt.subplots(figsize=(5, 5))
            df["Automation_Risk"].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=20, ax=ax)
            ax.set_title('Distribution of Automation Risk')
            st.pyplot(fig)

        elif analysis_option == "Automation Risk Analysis":
            st.write("### Automation Risk Analysis")
            st.write("#### By AI Adoption Level")
            st.write(df.groupby(["Automation_Risk", "AI_Adoption_Level"]).size().unstack())
            st.write("#### By Job Title")
            st.write(df.groupby(["Automation_Risk", "Job_Title"]).size().unstack())
            st.write("#### By Industry")
            st.write(df.groupby(["Automation_Risk", "Industry"]).size().unstack())
            st.write("#### By Location")
            st.write(df.groupby(["Automation_Risk", "Location"]).size().unstack())

        elif analysis_option == "Required Skills Analysis":
            st.write("### Required Skills Analysis")
            st.write(df['Required_Skills'].value_counts())
            st.write("#### By Job Title")
            st.write(df.groupby(["Required_Skills", "Job_Title"]).size().unstack())
            st.write("#### By Industry")
            st.write(pd.crosstab(index=df['Industry'], columns=[df['Required_Skills']]))

        elif analysis_option == "Salary Distribution":
            st.write("### Salary Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Salary_USD'].plot(kind='hist', bins=20, color='lightblue', edgecolor='black', ax=ax)
            ax.set_title('Salary Distribution')
            ax.set_xlabel('Salary (USD)')
            st.pyplot(fig)

        elif analysis_option == "Salary by Industry and AI Adoption Level":
            st.write("### Salary by Industry and AI Adoption Level")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=df, x='Industry', y='Salary_USD', hue='AI_Adoption_Level', marker='o', ax=ax)
            ax.set_title('Salary Across Different Industries with AI Adoption Level')
            ax.set_xlabel('Industry')
            ax.set_ylabel('Salary (USD)')
            ax.legend(title='AI Adoption Level', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "Salary by Location and Automation Risk":
            st.write("### Salary by Location and Automation Risk")
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.lineplot(data=df, x='Location', y='Salary_USD', hue='Automation_Risk', marker='o', ax=ax)
            ax.set_title('Salary Trends Across Locations by Automation Risk')
            ax.set_xlabel('Location')
            ax.set_ylabel('Salary (USD)')
            ax.legend(title='Automation Risk', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "Salary by Job Title and Automation Risk":
            st.write("### Salary by Job Title and Automation Risk")
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.lineplot(data=df, x='Job_Title', y='Salary_USD', hue='Automation_Risk', marker='o', ax=ax)
            ax.set_title('Salary Trends Across Job Titles by Automation Risk')
            ax.set_xlabel('Job Title')
            ax.set_ylabel('Salary (USD)')
            ax.legend(title='Automation Risk', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        elif analysis_option == "Remote Work Analysis":
            st.write("### Remote Work Analysis")
            # Check if 'Remote_Friendly' column exists
            if 'Remote_Friendly' in df.columns:
                fig, ax = plt.subplots(figsize=(5, 5))
                df["Remote_Friendly"].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=20, ax=ax)
                ax.set_title('Distribution of Remote Work')
                st.pyplot(fig)
            else:
                st.error("Column 'Remote_Friendly' not found in the dataset.")

        elif analysis_option == "Job Growth Projection Analysis":
            st.write("### Job Growth Projection Analysis")
            st.write("#### Job Growth Projection Counts")
            st.write(df['Job_Growth_Projection'].value_counts())
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, x='Job_Growth_Projection', hue='Industry', ax=ax)
            ax.set_title('Job Growth Projection Across Different Industries')
            ax.set_xlabel('Job Growth Projection')
            ax.set_ylabel('Count')
            ax.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
