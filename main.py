import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load and cache the model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_regressor_model.pkl')

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('/content/train.csv')
    return data

# Define a function for model prediction and evaluation
def evaluate_model(X_train, X_test, y_train, y_test):
    model = load_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Main function to run the app
def main():
    st.title("Sales Prediction App")
    
    # Sidebar for user options
    st.sidebar.header("App Configuration")
    if st.sidebar.checkbox("Show raw data"):
        data = load_data()
        st.subheader("Raw Data")
        st.write(data.head())
        
    # Data loading and preprocessing
    data = load_data()
    data['Postal Code'].fillna(data['Postal Code'].mean(), inplace=True)
    
    # Splitting data
    features = data.drop(columns=['Sales', 'Country'])  # Drop unnecessary columns
    target = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Display training progress
    with st.spinner("Training the model..."):
        y_pred = evaluate_model(X_train, X_test, y_train, y_test)
        st.success("Model training complete!")

    # Display performance metrics
    st.header("Model Evaluation Metrics")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

    # Visualizations in container
    st.header("Data Visualization")
    with st.container():
        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt)

        st.subheader("Feature Importances")
        model = load_model()
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        feature_names = X_train.columns
        sorted_indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_indices], align='center')
        plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=90)
        plt.title("Feature Importance")
        st.pyplot(plt)

    # User input for prediction
    st.sidebar.header("Predict Sales")
    st.sidebar.write("Enter input values for prediction:")
    row_id = st.sidebar.number_input("Row ID", min_value=0, step=1)
    order_id = st.sidebar.text_input("Order ID", "CA-2017-152156")
    order_date = st.sidebar.date_input("Order Date")
    ship_date = st.sidebar.date_input("Ship Date")
    ship_mode = st.sidebar.selectbox("Ship Mode", data['Ship Mode'].unique())
    customer_id = st.sidebar.text_input("Customer ID", "CG-12520")
    customer_name = st.sidebar.text_input("Customer Name", "Claire Gute")
    segment = st.sidebar.selectbox("Segment", data['Segment'].unique())
    city = st.sidebar.text_input("City", "Henderson")
    state = st.sidebar.text_input("State", "Kentucky")
    postal_code = st.sidebar.number_input("Postal Code", value=42420)
    region = st.sidebar.selectbox("Region", data['Region'].unique())
    product_id = st.sidebar.text_input("Product ID", "FUR-BO-10001798")
    category = st.sidebar.selectbox("Category", data['Category'].unique())
    sub_category = st.sidebar.selectbox("Sub-Category", data['Sub-Category'].unique())
    product_name = st.sidebar.text_input("Product Name", "Bush Somerset Collection Bookcase")

    # Organize input into DataFrame for prediction
    user_data = pd.DataFrame({
        'Row ID': [row_id], 'Order ID': [order_id], 'Order Date': [order_date],
        'Ship Date': [ship_date], 'Ship Mode': [ship_mode], 'Customer ID': [customer_id],
        'Customer Name': [customer_name], 'Segment': [segment], 'City': [city], 'State': [state],
        'Postal Code': [postal_code], 'Region': [region], 'Product ID': [product_id],
        'Category': [category], 'Sub-Category': [sub_category], 'Product Name': [product_name]
    })

    if st.sidebar.button("Predict Sales"):
        # Transform user data if necessary (e.g., encoding)
        st.write("User input data:", user_data)
        prediction = model.predict(user_data)
        st.write(f"**Predicted Sales:** ${prediction[0]:.2f}")

# Run the app
if __name__ == "__main__":
    main()
