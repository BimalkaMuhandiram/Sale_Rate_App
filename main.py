# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
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
    train_data['order_date'] = pd.to_datetime(train_data['order_date']).dt.date
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
rpath = "path/to/your/dataset.csv"  # Update this to your dataset path
train_data = load_data(rpath)

# Loading the toolkit
loaded_toolkit = load_ml_toolkit("path/to/your/ml_toolkit.pkl")  # Update this to your model path
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
    image = Image.open("path/to/your/image.jpg")  # Update this to your image path
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
                    - **ship_mode**: Shipping method.
                    - **customer_id**: Unique identifier for the customer.
                    - **customer_name**: Name of the customer.
                    - **segment**: Customer segment (e.g., consumer, corporate).
                    - **country**: Country of the customer.
                    - **city**: City of the customer.
                    - **state**: State of the customer.
                    - **postal_code**: Postal code of the customer.
                    - **region**: Region of the customer.
                    - **product_id**: Unique identifier for the product.
                    - **category**: Product category.
                    - **sub-category**: Product sub-category.
                    - **product_name**: Name of the product.
                    - **sales**: Total sales amount for the order.
                    """)

# Structuring the dataset section
with dataset:
    dataset.markdown("**This is the dataset of Corporation Favorita**")
    check = dataset.checkbox("Preview the dataset")
    if check:
        dataset.write(train_data.head())
    dataset.write("View sidebar for information on the columns")

# Defining the list of expected variables
expected_inputs = ["order_date", "customer_id", "customer_name", "segment", "country", "city", 
                   "state", "postal_code", "region", "product_id", "category", "sub-category", 
                   "product_name", "ship_mode"]

# List of features to encode
categoricals = ["customer_id", "customer_name", "segment", "country", "city", "state", 
                "postal_code", "region", "product_id", "category", "sub-category", "product_name", "ship_mode"]

# List of features to scale
cols_to_scale = []

with features_and_output:
    features_and_output.subheader("Give us your Inputs")
    features_and_output.write("This section captures your input to be used in predictions")

    col1, col2 = features_and_output.columns(2)

    # Designing the input section of the app
    with form:
        order_date = col1.date_input("Select an order date:", min_value=train_data["order_date"].min())
        customer_id = col1.selectbox("Customer ID:", options=(list(train_data['customer_id'].unique())))
        customer_name = col1.text_input("Customer Name:", value="")
        segment = col1.selectbox("Segment:", options=(train_data['segment'].unique()))
        country = col1.selectbox("Country:", options=(train_data['country'].unique()))
        city = col1.selectbox("City:", options=(train_data['city'].unique()))
        state = col1.selectbox("State:", options=(train_data['state'].unique()))
        postal_code = col1.text_input("Postal Code:", value="")
        region = col1.selectbox("Region:", options=(train_data['region'].unique()))
        product_id = col2.selectbox("Product ID:", options=(train_data['product_id'].unique()))
        category = col2.selectbox("Category:", options=(train_data['category'].unique()))
        sub_category = col2.selectbox("Sub-category:", options=(train_data['sub-category'].unique()))
        product_name = col2.text_input("Product Name:", value="")
        ship_mode = col2.selectbox("Ship Mode:", options=(train_data['ship_mode'].unique()))

        # Submit button
        submitted = form.form_submit_button(label="Submit")

    if submitted:
        st.success('All Done!', icon="✅")  
        
        # Inputs formatting
        input_dict = {
            "order_date": [order_date],
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
            "product_name": [product_name],
            "ship_mode": [ship_mode]
        }

        # Converting the input into a dataframe
        input_data = pd.DataFrame.from_dict(input_dict)
        input_df = input_data.copy()
        
        # Converting data types into required types
        input_data["order_date"] = pd.to_datetime(input_data["order_date"]).dt.date
        
        # Getting date features
        df_processed = getDateFeatures(input_data, "order_date")

        # Encoding the categoricals
        encoded_categoricals = encode.transform(input_data[categoricals])
        encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encode.get_feature_names_out().tolist())
        df_processed = df_processed.join(encoded_categoricals)
        df_processed.drop(columns=categoricals, inplace=True)
        df_processed.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace=True)

        # Making the predictions        
        dt_pred = ml_model.predict(df_processed)
        df_processed["sales"] = dt_pred
        input_df["sales"] = dt_pred
        display = dt_pred[0]

        # Adding the predictions to previous predictions
        st.session_state["results"].append(input_df)
        result = pd.concat(st.session_state["results"])

        # Displaying prediction results
        st.success(f"**Predicted sales**: USD {display}")
