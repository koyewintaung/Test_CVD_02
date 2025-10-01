import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data(file_path, file_type):
    if file_type == "csv":
        data = pd.read_csv(file_path)
    elif file_type == "xlsx":
        data = pd.read_excel(file_path)
    return data

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    data = load_data(uploaded_file, file_type)
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Descriptive Statistics
    st.write("### Descriptive Statistics")
    st.dataframe(data.describe())

    # Data Visualization
    st.write("### Data Visualization")

    # Choose columns for histogram
    hist_columns = st.multiselect("Select columns for histogram", data.columns)

    if hist_columns:
        st.write("#### Histograms")
        for column in hist_columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)

    # Choose columns for scatter plot
    st.write("#### Scatter Plots")
    x_column = st.selectbox("Select X axis", data.columns)
    y_column = st.selectbox("Select Y axis", data.columns)

    if x_column and y_column:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)

    # Correlation heatmap
    st.write("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
    st.pyplot(fig)

    # Basic Data Analysis
    st.write("### Basic Data Analysis")

    # Display value counts for categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    for column in categorical_cols:
        st.write(f"#### Value Counts for {column}")
        st.dataframe(data[column].value_counts())

    # Simple Machine Learning Model
    st.write("### Simple Machine Learning Model")

    # Prepare data for machine learning
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"#### Accuracy: {accuracy}")

else:
    st.warning("Please upload a CSV or Excel file to proceed.")
