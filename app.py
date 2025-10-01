import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # For interactive charts like sunkey and more
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

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
    st.write("### Data Types")
    st.write(data.dtypes)

    # Handle non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    st.write("Non-numeric columns:", non_numeric_cols)

    # Option 1: Remove non-numeric columns
    data_numeric = data.drop(non_numeric_cols, axis=1, errors='ignore')

    # Option 2: Try to convert non-numeric columns to numeric
    # for col in non_numeric_cols:
    #     try:
    #         data[col] = pd.to_numeric(data[col])
    #     except ValueError:
    #         st.write(f"Could not convert column {col} to numeric.")

    # Handle missing values
    data_numeric = data_numeric.fillna(0)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(data_numeric.corr(), annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error creating heatmap: {e}")

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
    if 'target' in data.columns:
        X = data.drop('target', axis=1, errors='ignore')
        y = data['target']

        # Ensure numeric data types in X
        X = X.select_dtypes(include=['number'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Remove infinite values
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill remaining NaN values with the mean
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)

        # Train a Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"#### Accuracy: {accuracy}")
    else:
        st.write("Target column 'target' not found in the dataset.")

    # Additional Charts
    st.write("### Additional Charts")
    chart_type = st.selectbox("Select chart type",
                              ["Bar Chart", "Pie Chart", "Sunkey Diagram",
                               "100% Stacked Bar Chart", "Stacked Vertical Bar Chart",
                               "Line Chart"])

    st.write(f"Selected chart type: {chart_type}")  # Debugging line

    if chart_type == "Bar Chart":
        column_x = st.selectbox("Select X axis for Bar Chart", data.columns)
        column_y = st.selectbox("Select Y axis for Bar Chart", data.columns)
        st.write(f"Selected X column: {column_x}")  # Debugging line
        st.write(f"Selected Y column: {column_y}")  # Debugging line
        if column_x and column_y:
            fig = px.bar(data, x=column_x, y=column_y)
            st.plotly_chart(fig)

    elif chart_type == "Pie Chart":
        column = st.selectbox("Select column for Pie Chart", data.columns)
        st.write(f"Selected column: {column}")  # Debugging line
        if column:
            fig = px.pie(data, names=column)
            st.plotly_chart(fig)

    elif chart_type == "Sunkey Diagram":
        source = st.selectbox("Select source column for Sunkey Diagram", data.columns)
        target = st.selectbox("Select target column for Sunkey Diagram", data.columns)
        values = st.selectbox("Select values column for Sunkey Diagram", data.columns)
        st.write(f"Selected source column: {source}")  # Debugging line
        st.write(f"Selected target column: {target}")  # Debugging line
        st.write(f"Selected values column: {values}")  # Debugging line
        if source and target and values:
            fig = px.sunburst(data, path=[source, target], values=values)
            st.plotly_chart(fig)

    elif chart_type == "100% Stacked Bar Chart":
        column_x = st.selectbox("Select X axis for 100% Stacked Bar Chart", data.columns)
        column_y = st.selectbox("Select Y axis for 100% Stacked Bar Chart", data.columns)
        color = st.selectbox("Select color column for 100% Stacked Bar Chart", data.columns)
        st.write(f"Selected X column: {column_x}")  # Debugging line
        st.write(f"Selected Y column: {column_y}")  # Debugging line
        st.write(f"Selected color column: {color}")  # Debugging line
        if column_x and column_y and color:
            # Group data and calculate percentages
            grouped = data.groupby([column_x, color])[column_y].sum().unstack().fillna(0)
            grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
            fig = px.bar(grouped, x=grouped.index, y=grouped.columns,
                         labels={'value': 'Percentage'},
                         title='100% Stacked Bar Chart')
            st.plotly_chart(fig)

    elif chart_type == "Stacked Vertical Bar Chart":
        column_x = st.selectbox("Select X axis for Stacked Vertical Bar Chart", data.columns)
        column_y = st.selectbox("Select Y axis for Stacked Vertical Bar Chart", data.columns)
        color = st.selectbox("Select color column for Stacked Vertical Bar Chart", data.columns)
        st.write(f"Selected X column: {column_x}")  # Debugging line
        st.write(f"Selected Y column: {column_y}")  # Debugging line
        st.write(f"Selected color column: {color}")  # Debugging line
        if column_x and column_y and color:
            fig = px.bar(data, x=column_x, y=column_y, color=color,
                         title='Stacked Vertical Bar Chart')
            st.plotly_chart(fig)

    elif chart_type == "Line Chart":
        column_x = st.selectbox("Select X axis for Line Chart", data.columns)
        column_y = st.selectbox("Select Y axis for Line Chart", data.columns)
        st.write(f"Selected X column: {column_x}")  # Debugging line
        st.write(f"Selected Y column: {column_y}")  # Debugging line
        if column_x and column_y:
            fig = px.line(data, x=column_x, y=column_y,
                          title='Line Chart')
            st.plotly_chart(fig)

else:
    st.warning("Please upload a CSV or Excel file to proceed.")
