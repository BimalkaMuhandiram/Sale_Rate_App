import streamlit as st
import joblib

# Load the trained model
model = joblib.load('random_forest_regressor_model.pkl')

# Streamlit app
st.title("Sales Prediction App")

# Input fields for features
row_id = st.text_input("Row ID", value="1")
order_id = st.text_input("Order ID")
order_date = st.date_input("Order Date")
ship_date = st.date_input("Ship Date")
ship_mode = st.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
customer_id = st.text_input("Customer ID")
customer_name = st.text_input("Customer Name")
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
city = st.text_input("City")
state = st.text_input("State")
postal_code = st.number_input("Postal Code", min_value=0)
region = st.selectbox("Region", ["East", "West", "Central", "South"])
category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
sub_category = st.text_input("Sub-Category")
product_id = st.text_input("Product ID")
product_name = st.text_input("Product Name")

# When the user clicks "Predict", create a DataFrame and make a prediction
if st.button("Predict"):
    # Create a DataFrame from user inputs
    input_data = {
        'row_id': row_id,
        'order_id': order_id,
        'order_date': order_date,
        'ship_date': ship_date,
        'ship_mode': ship_mode,
        'customer_id': customer_id,
        'customer_name': customer_name,
        'segment': segment,
        'city': city,
        'state': state,
        'postal_code': postal_code,
        'region': region,
        'category': category,
        'sub-category': sub_category,
        'product_id': product_id,
        'product_name': product_name
    }

    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make a prediction
    try:
        prediction = model.predict(input_df)
        st.write(f"Predicted Sales: ${prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
    except Exception as ex:
        st.error(f"An unexpected error occurred: {ex}")
