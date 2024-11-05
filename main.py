import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the app title
st.title("Sales Prediction App")

# Load data function with Streamlit's file_uploader
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.error("Please upload a valid CSV file.")
        return None

# Sidebar for file upload
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load data only if file is uploaded
data = load_data(uploaded_file)

# Main function for data processing and visualization
def main():
    if data is not None:
        st.write("## Data Preview")
        st.write(data.head())

        # Display basic statistics
        st.write("## Data Summary")
        st.write(data.describe())

        # Plotting section - Example plot
        st.write("## Sales Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Sales'], kde=True)
        st.pyplot(plt)

        # Additional data visualizations
        # Add any other visualizations or data processing here as needed
    else:
        st.info("Awaiting CSV file upload.")

if __name__ == "__main__":
    main()
