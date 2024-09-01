import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
Discover which classifier works best on different datasets!
""")

# Option to upload user's own dataset
uploaded_file = st.sidebar.file_uploader("Upload your own dataset (CSV file)", type=["csv"])

# Dataset selection in the sidebar
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine', 'Digits', 'Boston') if uploaded_file is None else ('User Uploaded Dataset',)
)

st.write(f"### {dataset_name} Dataset")

# Classifier selection in the sidebar
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Gradient Boosting')
)

# Function to load the user's uploaded dataset
def load_user_dataset(file):
    data = pd.read_csv(file)
    X = data.iloc[:, :-1].values  # Features (all columns except the last one)
    y = data.iloc[:, -1].values   # Target (last column)
    return X, y

# Function to get the sklearn dataset based on user selection
def get_sklearn_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    elif name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    elif name == 'Digits':
        data = datasets.load_digits()
    elif name == 'Boston':
        data = datasets.load_boston()
    X = data.data
    y = data.target
    return X, y

# Decide which dataset to use: user's uploaded or predefined sklearn dataset
if uploaded_file is not None:
    X, y = load_user_dataset(uploaded_file)
else:
    X, y = get_sklearn_dataset(dataset_name)

st.write('**Shape of dataset:**', X.shape)
st.write('**Number of classes:**', len(np.unique(y)))

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

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

# Plotting the dataset using PCA
st.subheader('Dataset Visualization (PCA)')
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig2 = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(fig2)

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
