import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from datetime import datetime
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_regressor_model.pkl")

model = load_model()

# App Title
st.title("Sales Prediction App")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

def user_input_features():
    # Collecting user input based on relevant features in the dataset
    order_quantity = st.sidebar.slider("Order Quantity", 1, 100, 10)  # Assume this is a numerical feature
    discount = st.sidebar.slider("Discount (%)", 0, 100, 10)  # Discount percentage
    customer_id = st.sidebar.text_input("Customer ID", "CUST_001")  # Example input for customer ID
    customer_name = st.sidebar.text_input("Customer Name", "John Doe")  # Example input for customer name
    category = st.sidebar.selectbox("Product Category", ["Furniture", "Office Supplies", "Technology"])  # Example categorical feature
    city = st.sidebar.text_input("City", "New York")  # Example input for city
    order_date = st.sidebar.date_input("Order Date", value=datetime.today())  # Default to today

    # Create a DataFrame with user inputs
    data = {
        'order_quantity': order_quantity,
        'discount': discount,
        'customer_id': customer_id,
        'customer_name': customer_name,
        'category': category,
        'city': city,
        'order_date': order_date
    }
    features = pd.DataFrame(data, index=[0])

    return features

input_data = user_input_features()

# Progress bar for prediction
if st.button("Predict"):
    with st.spinner("Making prediction..."):
        try:
            # Prepare the input features for prediction
            categorical_features = ['customer_id', 'customer_name', 'category', 'city']
            numerical_features = ['order_quantity', 'discount']

            # One-Hot Encoding for categorical features
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoded_categorical = encoder.fit_transform(input_data[categorical_features]).toarray()
            encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))
            
            # Combine encoded categorical features with numerical features
            final_input = pd.concat([input_data[numerical_features], encoded_df], axis=1)

            # Make prediction
            prediction = model.predict(final_input)
            st.success(f"Predicted Sales: ${prediction[0]:.2f}")
        except ValueError as e:
            st.error(f"Prediction Error: {str(e)}")  # Show error if prediction fails

# Media Upload Section
st.header("Upload Your Media")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
uploaded_csv = st.file_uploader("Upload your CSV file...", type=["csv"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)  # Open the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error: {e}")  # Show error if image cannot be opened

if uploaded_csv is not None:
    try:
        data = pd.read_csv(uploaded_csv)  # Read the uploaded CSV file
        st.write("Data from CSV:")
        st.dataframe(data)  # Display the dataframe in the app

        # Display basic statistics about the data
        st.write("Basic Statistics:")
        st.write(data.describe())

        # Example: Display sales by category
        st.subheader("Sales by Category")
        sales_by_category = data.groupby('category')['sales'].sum().reset_index()
        st.bar_chart(sales_by_category.set_index('category'))

        # Example: Display sales over time
        st.subheader("Sales Over Time")
        data['order_date'] = pd.to_datetime(data['order_date'])  # Ensure the date column is in datetime format
        sales_over_time = data.groupby(data['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
        sales_over_time['order_date'] = sales_over_time['order_date'].dt.to_timestamp()  # Convert back to timestamp for plotting
        st.line_chart(sales_over_time.set_index('order_date')['sales'])

    except Exception as e:
        st.error(f"Error loading CSV: {e}")  # Show error if CSV cannot be read
