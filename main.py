import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from datetime import datetime

# Function to load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("train.csv") 

# Load data
data = load_data()

# Home Page
def home_page():
    st.title("Sales Prediction App")
    
    # Upload image option
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.write("""
    Welcome to the Sales Prediction App! Here, you can predict sales based on various features.
    Use the sidebar to navigate through the different pages and predict sales using our machine learning model.
    """)

    # Dataset Overview
    st.subheader('Dataset Overview')
    st.dataframe(data.head())
    st.write("Summary of dataset:")
    st.write(data.describe())

    # Visualizing sales distribution
    st.subheader('Sales Distribution')
    fig, ax = plt.subplots()
    sns.histplot(data['Sales'], kde=True, bins=20, ax=ax)
    ax.set_title('Sales Distribution')
    ax.set_xlabel('Sales')
    st.pyplot(fig)

    # Customer Segment Distribution
    if 'Segment' in data.columns:
        segment_counts = data['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig = px.pie(segment_counts, values='Count', names='Segment', title='Customer Segment Distribution')
        st.plotly_chart(fig)

# Model Training Page
def model_page():
    st.title("Sales Prediction Model")
    
    # Sidebar for user inputs
    st.sidebar.header("Model Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
    max_depth = st.sidebar.slider("Maximum Depth", 1, 20, 10)
    random_state = st.sidebar.number_input("Random State", value=42)

    # Preprocessing the Data
    st.subheader("Preprocessing the Data")
    
    # Features and target variable
    X = data.drop('Sales', axis=1, errors='ignore')  # Features
    y = data['Sales']  # Target

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Display feature and target summaries
    with st.expander("Feature Summary"):
        st.write(X.describe(include='all'))

    with st.expander("Target Summary"):
        st.write(y.describe())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Define preprocessing for numerical and categorical features
    numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),  # Keep numeric features unchanged
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical features
        ])

    # Train the model
    st.subheader("Training the Model")
    train_button = st.button("Train Model")

    if train_button:
        with st.spinner('Training in progress...'):
            try:
                # Create a pipeline that first transforms the data and then fits the model
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', RandomForestRegressor(n_estimators=n_estimators, 
                                                                             max_depth=max_depth, 
                                                                             random_state=random_state))])
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

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Prediction Page
def prediction_page():
    st.title("Sales Prediction")

    loaded_model = joblib.load(open('sales_model.joblib', 'rb'))  # Load your trained model

    def sales_prediction(input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        return loaded_model.predict(input_data_reshaped)

    st.markdown("### Enter your details for sales prediction")
    order_quantity = st.number_input('Order Quantity', min_value=0, help='Enter the quantity of the order')
    discount = st.number_input('Discount (%)', min_value=0.0, max_value=100.0, step=0.1, help='Enter the discount percentage')
    category = st.selectbox('Product Category', ['Furniture', 'Office Supplies', 'Technology'])
    city = st.text_input('City', help='Enter the city name')
    customer_id = st.text_input('Customer ID', help='Enter the customer ID')
    customer_name = st.text_input('Customer Name', help='Enter the customer name')
    order_date = st.date_input("Order Date", value=datetime.today())

    if st.button('Click to Predict Sales'):
        with st.spinner('Calculating...'):
            input_data = [order_quantity, discount, category, city, customer_id, customer_name]
            prediction = sales_prediction(input_data)
            st.success(f"Predicted Sales: ${prediction[0]:.2f}")

# Sidebar Navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Home", "Sales Prediction Model", "Prediction"])

# Page Navigation Logic
if page_selection == "Home":
    home_page()
elif page_selection == "Sales Prediction Model":
    model_page()
elif page_selection == "Prediction":
    prediction_page()
