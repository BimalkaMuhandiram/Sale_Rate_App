import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load your dataset
data = pd.read_csv('your_data_file.csv')  # Replace with your actual data file

# Define feature set X and target variable y
X = data[['order_date', 'ship_date', 'ship_mode', 'segment', 'city', 'state',
           'postal_code', 'region', 'category', 'sub-category', 'product_id', 'product_name']]
y = data['sales']  # Target variable

# Preprocessing function
def preprocess_data(input_data):
    # Normalize column names to lowercase
    input_data.columns = input_data.columns.str.lower()
    
    # Convert dates to datetime objects
    input_data['order_date'] = pd.to_datetime(input_data['order_date'])
    input_data['ship_date'] = pd.to_datetime(input_data['ship_date'])

    # Encode categorical columns
    categorical_columns = ['ship_mode', 'segment', 'region', 'category', 'sub-category']
    le = LabelEncoder()
    
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna("Unknown")  # Fill missing values
            input_data[col] = le.fit_transform(input_data[col])  # Label encode categorical variables

    # Handle missing numeric values
    numeric_cols = input_data.select_dtypes(include=['float64', 'int64']).columns
    input_data[numeric_cols] = input_data[numeric_cols].fillna(0)  # Fill numeric missing values with 0

    return input_data

# Preprocess the data
X_processed = preprocess_data(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'random_forest_regressor_model.pkl')

# Example of making a prediction
# Prepare input for prediction (you can use similar input as in your Streamlit app)
example_input = pd.DataFrame({
    'order_date': pd.to_datetime(['2023-11-01']),
    'ship_date': pd.to_datetime(['2023-11-02']),
    'ship_mode': ['First Class'],
    'segment': ['Consumer'],
    'city': ['New York'],
    'state': ['NY'],
    'postal_code': [10001],
    'region': ['East'],
    'category': ['Furniture'],
    'sub-category': ['Chairs'],
    'product_id': ['FURN-001'],
    'product_name': ['Office Chair']
})

# Preprocess example input
example_input_processed = preprocess_data(example_input)

# Make a prediction
prediction = model.predict(example_input_processed)
print(f"Predicted Sales: ${prediction[0]:,.2f}")
