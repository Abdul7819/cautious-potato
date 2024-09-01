import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to load and preprocess data
@st.cache
def load_data():
    df = pd.read_csv("nearest-earth-objects(1910-2024).csv")
    df.dropna(inplace=True)
    features_list = ["absolute_magnitude", "estimated_diameter_min", "estimated_diameter_max", "relative_velocity"]
    for feature in features_list:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.65)
        iqr = q3 - q1
        upper_limit = q3 + (1.5 * iqr)
        lower_limit = q1 - (1.5 * iqr)
        df = df.loc[(df[feature] < upper_limit) & (df[feature] > lower_limit)]
    return df

# Load data
df = load_data()

# Streamlit UI
st.title('NEO Classification App')
st.sidebar.header('Model Selection')

# Model Selection
model_option = st.sidebar.selectbox(
    'Choose a Classification Model',
    ['Random Forest', 'XGBoost', 'LightGBM']
)

# Features and target variable
X = df.drop(["neo_id", "name", "orbiting_body", "is_hazardous"], axis=1)
y = df["is_hazardous"]

# Handle imbalance
somte = SMOTE(random_state=42)
X, y = somte.fit_resample(X, y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)

# Initialize classifier based on user selection
if model_option == 'Random Forest':
    clf = RandomForestClassifier(random_state=42)
elif model_option == 'XGBoost':
    clf = XGBClassifier(random_state=42)
else:
    clf = LGBMClassifier(random_state=42)

# Train model
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Display evaluation metrics
st.subheader('Model Evaluation')
st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix Heatmap')
st.pyplot(fig)

# Feature Importance
if model_option in ['Random Forest', 'XGBoost', 'LightGBM']:
    st.subheader('Feature Importance')
    feature_importances = clf.feature_importances_ if model_option != 'XGBoost' and model_option != 'LightGBM' else clf.feature_importances_
    feature_names = x_train.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='blue', height=0.4)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance')
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
