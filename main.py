import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from datetime import datetime

# Function to load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")  # Replace with the path to your dataset
    return data

# Load data
data = load_data()

# Define the home page
def home_page():
    st.title("Sales Prediction App")
    
    # Upload media (example images)
    st.image("sales_image.jpg")  # Replace with your image path
    st.video("sales_video.mp4")  # Replace with your video path
    
    st.write("""
    Welcome to the Sales Prediction App! Here, you can predict sales based on input features from your dataset.
    Use the sidebar to navigate through the different pages and to train the model or make predictions.
    """)

    # Display basic data overview
    st.subheader('Dataset Overview')
    st.write(data.head())
    st.write("Summary of dataset:")
    st.write(data.describe())

    # Visualizing sales distribution
    st.subheader('Sales Distribution')
    fig, ax = plt.subplots()
    sns.histplot(data['Sales'], kde=True, bins=20, ax=ax)
    ax.set_title('Sales Distribution')
    ax.set_xlabel('Sales')
    st.pyplot(fig)

    # Gender Distribution (if applicable, based on dataset)
    if 'Segment' in data.columns:
        segment_counts = data['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig = px.pie(segment_counts, values='Count', names='Segment', title='Customer Segment Distribution')
        st.plotly_chart(fig)

# Define the model training page
def model_page():
    st.title("Sales Prediction Model")
    
    # Sidebar for user inputs
    st.sidebar.header("Model Hyperparameters")

    # Input widgets for model parameters
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100, help="The number of trees in the forest.")
    max_depth = st.sidebar.slider("Maximum Depth", 1, 20, 10, help="The maximum depth of the trees.")
    random_state = st.sidebar.number_input("Random State", value=42, help="Random seed for reproducibility.")

    # Preprocessing
    st.subheader("Preprocessing the Data")
    X = data.drop('Sales', axis=1)  # Features
    y = data['Sales']  # Target

    # Display feature and target summaries
    with st.expander("Feature Summary"):
        st.write(X.describe())

    with st.expander("Target Summary"):
        st.write(y.describe())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train the model
    st.subheader("Training the Model")
    train_button = st.button("Train Model")

    if train_button:
        with st.spinner('Training in progress...'):
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            st.success(f"Model Trained! RMSE: *{rmse:.2f}*")

            # Visualize predictions vs actual values
            st.subheader("Predictions vs Actual")
            comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.line_chart(comparison_df)

# Define the prediction page
def prediction_page():
    st.title("Sales Prediction")

    loaded_model = joblib.load(open('sales_model.joblib', 'rb'))  # Load your trained model

    def sales_prediction(input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = loaded_model.predict(input_data_reshaped)
        return f"Predicted Sales: ${prediction[0]:.2f}"

    def main():
        st.markdown("### Enter your details")
        # User inputs based on your dataset
        order_quantity = st.text_input('Order Quantity')
        discount = st.text_input('Discount (%)')
        category = st.selectbox('Product Category', ['Furniture', 'Office Supplies', 'Technology'])  # Add more categories as needed
        city = st.text_input('City')
        customer_id = st.text_input('Customer ID')
        customer_name = st.text_input('Customer Name')
        order_date = st.date_input("Order Date", value=datetime.today())

        if st.button('Click to Predict Sales'):
            with st.spinner('Calculating...'):
                prediction = sales_prediction([order_quantity, discount, category, city, customer_id, customer_name])
                st.success(prediction)

    main()

# Sidebar Navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Home", "Sales Prediction Model", "Prediction"])

# Sidebar - Link to the different pages
if page_selection == "Home":
    home_page()
elif page_selection == "Sales Prediction Model":
    model_page()
elif page_selection == "Prediction":
    prediction_page()
