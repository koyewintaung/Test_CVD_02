import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()
