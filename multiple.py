import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Enhanced ML App", layout="wide", initial_sidebar_state="expanded")

# Title of the Streamlit app
st.title('Enhanced Machine Learning Explorer')

st.write("""
## Compare Different Classifiers and Datasets
Discover which classifier works best on your uploaded dataset!
""")

# Option to upload user's own dataset
uploaded_file = st.sidebar.file_uploader("Upload your own dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Function to load the user's uploaded dataset
    def load_user_dataset(file):
        data = pd.read_csv(file)
        
        # Check for non-numeric columns in features and convert if necessary
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Convert target variable to numeric if it's categorical
        if y.dtypes == 'object':
            y = pd.factorize(y)[0]  # Convert categories to numeric codes
        
        X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coercing errors
        
        # Drop rows with NaN values that result from the conversion
        X = X.dropna()
        
        # Convert DataFrame to numpy arrays for consistency with sklearn
        X = X.values
        y = y[:X.shape[0]]  # Ensure y matches the size of X after dropping rows
        
        return X, y

    X, y = load_user_dataset(uploaded_file)

    if X.size == 0 or y.size == 0:  # Check if dataset is empty
        st.error("The dataset is empty after preprocessing. Please check your data.")
    else:
        st.write(f"### Uploaded Dataset")

        # Display dataset information
        st.write('**Shape of dataset:**', X.shape)
        st.write('**Number of classes:**', len(np.unique(y)))

        # Classifier selection in the sidebar
        classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Gradient Boosting')
        )

        # Function to dynamically add classifier-specific parameters
        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == 'SVM':
                C = st.sidebar.slider('C (Regularization parameter)', 0.01, 10.0)
                kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                params['C'] = C
                params['kernel'] = kernel
            elif clf_name == 'KNN':
                K = st.sidebar.slider('K (Number of neighbors)', 1, 15)
                params['K'] = K
            elif clf_name == 'Random Forest':
                max_depth = st.sidebar.slider('max_depth', 2, 20)
                n_estimators = st.sidebar.slider('n_estimators', 10, 200)
                params['max_depth'] = max_depth
                params['n_estimators'] = n_estimators
            elif clf_name == 'Logistic Regression':
                C = st.sidebar.slider('C (Regularization parameter)', 0.01, 10.0)
                params['C'] = C
            elif clf_name == 'Gradient Boosting':
                learning_rate = st.sidebar.slider('Learning Rate', 0.01, 0.5)
                n_estimators = st.sidebar.slider('n_estimators', 50, 200)
                params['learning_rate'] = learning_rate
                params['n_estimators'] = n_estimators
            return params

        params = add_parameter_ui(classifier_name)

        # Function to create classifier objects based on user selection
        def get_classifier(clf_name, params):
            if clf_name == 'SVM':
                clf = SVC(C=params['C'], kernel=params['kernel'])
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            elif clf_name == 'Random Forest':
                clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                             max_depth=params['max_depth'], random_state=1234)
            elif clf_name == 'Logistic Regression':
                clf = LogisticRegression(C=params['C'], max_iter=1000)
            elif clf_name == 'Gradient Boosting':
                clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], 
                                                 learning_rate=params['learning_rate'], random_state=1234)
            return clf

        clf = get_classifier(classifier_name, params)

        # Splitting the dataset into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        except ValueError as e:
            st.error(f"Error splitting data: {e}")
        else:
            # Scaling the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training the classifier and making predictions
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Calculate accuracy
            acc = accuracy_score(y_test, y_pred)
            st.write(f'**Classifier:** {classifier_name}')
            st.write(f'**Accuracy:** {acc:.2f}')

            # Displaying classification report and confusion matrix
            st.subheader('Classification Report')
            st.text(classification_report(y_test, y_pred))

            st.subheader('Confusion Matrix')
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            # Visualization options for the user to select
            st.sidebar.subheader('Select Visualizations')
            visualize = st.sidebar.multiselect('Choose the graphs to display:', 
                                               ['Histogram', 'Scatter Plot', 'Box Plot', 'Correlation Matrix'])

            # Functions for different visualizations
            def plot_histogram(X, y):
                fig, ax = plt.subplots()
                ax.hist(y, bins=len(np.unique(y)), alpha=0.7, color='b')
                st.pyplot(fig)

            def plot_scatter(X, y):
                if X.shape[1] >= 2:  # Ensure there are at least 2 features for scatter plot
                    fig, ax = plt.subplots()
                    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                    st.pyplot(fig)
                else:
                    st.warning("Not enough features for scatter plot. Need at least 2 features.")

            def plot_boxplot(X):
                fig, ax = plt.subplots()
                ax.boxplot(X)
                st.pyplot(fig)

            def plot_correlation_matrix(data):
                fig, ax = plt.subplots()
                sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

            # Display selected visualizations
            if 'Histogram' in visualize:
                st.subheader('Histogram')
                plot_histogram(X, y)

            if 'Scatter Plot' in visualize:
                st.subheader('Scatter Plot')
                plot_scatter(X, y)

            if 'Box Plot' in visualize:
                st.subheader('Box Plot')
                plot_boxplot(X)

            if 'Correlation Matrix' in visualize:
                st.subheader('Correlation Matrix')
                plot_correlation_matrix(pd.DataFrame(X))

            # Adding custom styling to Streamlit components
            st.markdown("""
            <style>
                .reportview-container {
                    background: #F5F5F5;
                }
                .sidebar .sidebar-content {
                    background: #E0E0E0;
                }
            </style>
            """, unsafe_allow_html=True)

else:
    st.write("Please upload a dataset to start the analysis.")
