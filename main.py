# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Grocery Store Sales Prediction App", page_icon=":shopping_bags:", 
                   layout="wide", initial_sidebar_state="auto")

# Setting the page title
st.title("Sales Prediction")

# ---- Importing and creating other key elements items
# Importing machine learning toolkit
@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):
    with open(relative_path, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object

# Function to load the dataset
@st.cache()
def load_data(relative_path):
    train_data = pd.read_csv(relative_path, index_col=0)
    train_data["ship_date"] = pd.to_datetime(train_data["ship_date"]).dt.date
    train_data["order_date"] = pd.to_datetime(train_data["order_date"]).dt.date
    return train_data

# Function to get date features from the inputs
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df['is_weekend'] = np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    df = df.drop(columns="date")
    return df

# ----- Loading the key components
# Loading the base dataframe
rpath = "/Users/emmanythedon/Documents/train.csv"
train_data = load_data(rpath)

# Loading the toolkit
loaded_toolkit = load_ml_toolkit("/Users/emmanythedon/Documents/PostBAP_ASSESSMENT/ML_items")
if "results" not in st.session_state:
    st.session_state["results"] = []

# Instantiating the elements of the Machine Learning Toolkit
mscaler = loaded_toolkit["scaler"]
ml_model = loaded_toolkit["model"]
encode = loaded_toolkit["encoder"]

# Defining the base containers/ main sections of the app
header = st.container()
dataset = st.container()
features_and_output = st.container()

form = st.form(key="information", clear_on_submit=True)

# Structuring the header section
with header:
    # Icon for the page
    image = Image.open("/Users/emmanythedon/Documents/SALES FORECASTING/sales.jpg")
    st.image(image, width=500)

# Instantiating the form to receive inputs from the user
st.sidebar.header("This app predicts the sales of the Corporation Favorita grocery store")
check = st.sidebar.checkbox("Click here to know more about your columns")
if check:
    st.sidebar.markdown(""" 
                    - **row_id**: Unique identifier for each row.
                    - **order_id**: Unique identifier for each order.
                    - **order_date**: Date when the order was placed.
                    - **ship_date**: Date when the order was shipped.
                    - **ship_mode**: The mode of shipping.
                    - **customer_id**: Unique identifier for the customer.
                    - **customer_name**: Name of the customer.
                    - **segment**: Market segment of the customer.
                    - **country**: Country where the order was placed.
                    - **city**: City where the order was placed.
                    - **state**: State where the order was placed.
                    - **postal_code**: Postal code of the delivery address.
                    - **region**: Region where the order was placed.
                    - **product_id**: Unique identifier for the product.
                    - **category**: Category of the product.
                    - **sub-category**: Sub-category of the product.
                    - **product_name**: Name of the product.
                    - **sales**: Total sales value for the order.
                    """)

# Structuring the dataset section
with dataset:
    dataset.markdown("**This is the dataset of Corporation Favorita**")
    check = dataset.checkbox("Preview the dataset")
    if check:
        dataset.write(train_data.head())
    dataset.write("View sidebar for information on the columns")

# Defining the list of expected variables
expected_inputs = ["order_date", "ship_date", "ship_mode", "customer_id", "customer_name", 
                   "segment", "country", "city", "state", "postal_code", "region", 
                   "product_id", "category", "sub-category", "product_name", "sales"]

# List of features to encode
categoricals = ["ship_mode", "customer_id", "segment", "country", "city", "state", "region", "category", "sub-category", "product_name"]

# List of features to scale (if applicable)
cols_to_scale = []  # Add columns to scale if necessary

with features_and_output:
    features_and_output.subheader("Give us your Inputs")
    features_and_output.write("This section captures your input to be used in predictions")

    col1, col2 = features_and_output.columns(2)

    # Designing the input section of the app
    with form:
        order_date = col1.date_input("Select order date:", min_value=train_data["order_date"].min())
        ship_date = col1.date_input("Select ship date:", min_value=train_data["ship_date"].min())
        ship_mode = col1.selectbox("Ship mode:", options=train_data['ship_mode'].unique())
        customer_id = col1.selectbox("Customer ID:", options=train_data['customer_id'].unique())
        customer_name = col1.text_input("Customer Name:", "")
        segment = col1.selectbox("Segment:", options=train_data['segment'].unique())
        country = col1.selectbox("Country:", options=train_data['country'].unique())
        city = col1.selectbox("City:", options=train_data['city'].unique())
        state = col1.selectbox("State:", options=train_data['state'].unique())
        postal_code = col1.text_input("Postal Code:", "")
        region = col1.selectbox("Region:", options=train_data['region'].unique())
        product_id = col2.selectbox("Product ID:", options=train_data['product_id'].unique())
        category = col2.selectbox("Category:", options=train_data['category'].unique())
        sub_category = col2.selectbox("Sub-category:", options=train_data['sub-category'].unique())
        
        # Submit button
        submitted = form.form_submit_button(label="Submit")

    if submitted:
        st.success('All Done!', icon="✅")  
        
        # Inputs formatting
        input_dict = {
            "order_date": [order_date],
            "ship_date": [ship_date],
            "ship_mode": [ship_mode],
            "customer_id": [customer_id],
            "customer_name": [customer_name],
            "segment": [segment],
            "country": [country],
            "city": [city],
            "state": [state],
            "postal_code": [postal_code],
            "region": [region],
            "product_id": [product_id],
            "category": [category],
            "sub-category": [sub_category],
        }

        # Converting the input into a dataframe
        input_data = pd.DataFrame.from_dict(input_dict)
        input_df = input_data.copy()

        # Getting date features
        df_processed = getDateFeatures(input_data, "order_date")
        df_processed = getDateFeatures(df_processed, "ship_date")

        # Encoding the categoricals
        encoded_categoricals = encode.transform(input_data[categoricals])
        encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encode.get_feature_names_out().tolist())
        df_processed = df_processed.join(encoded_categoricals)
        df_processed.drop(columns=categoricals, inplace=True)
        df_processed.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace=True)

        # Making the predictions
        dt_pred = ml_model.predict(df_processed)
        df_processed["predicted_sales"] = dt_pred

        # Displaying prediction results
        st.success(f"**Predicted sales**: USD {dt_pred[0]:.2f}")

        # Adding the predictions to previous predictions
        st.session_state["results"].append(input_df)
        result = pd.concat(st.session_state["results"])

        # Expander to display previous predictions
        previous_output = st.expander("**Review previous predictions**")
        previous_output.data
