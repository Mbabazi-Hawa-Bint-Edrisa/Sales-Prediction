import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv(r'C:\Users\USER\Desktop\sales pred\Train.csv')
print("Dataset loaded:", data.head())

# Check for missing values
print("Missing values in dataset:\n", data.isna().sum())

# Define features and target
features = ['Item_Weight', 'Item_Fat_Content', 'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Location', 'Outlet_Type']
target = 'Item_Outlet_Sales'

# Preprocess the data
data['Outlet_Age'] = 2025 - data['Outlet_Establishment_Year']
data['Item_MRP_log'] = np.log1p(data['Item_MRP'])  # Log transform Item_MRP
data['Item_Outlet_Sales_log'] = np.log1p(data[target])  # Log transform target
data['Is_High_Fat'] = data['Item_Fat_Content'].apply(lambda x: 1 if x == 'Regular' else 0)

# Update features list with derived features
features = ['Item_Weight', 'Item_MRP_log', 'Outlet_Age', 'Is_High_Fat', 'Item_Fat_Content', 'Item_Type', 'Outlet_Location', 'Outlet_Type']
cat_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location', 'Outlet_Type']
num_features = ['Item_Weight', 'Item_MRP_log', 'Outlet_Age', 'Is_High_Fat']

# Prepare X and y
X = data[features]
y = data['Item_Outlet_Sales_log']

# Create a preprocessor with imputation for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute NaNs with mean
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

# Create a pipeline with preprocessor and Linear Regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Fit the pipeline
pipeline.fit(X, y)

# Calculate R² score
r2_score = pipeline.score(X, y)
print(f"R² Score: {r2_score}")

# Save the entire pipeline (preprocessor + model) to model.pkl
joblib.dump(pipeline, r'C:\Users\USER\Desktop\sales pred\artifacts\model.pkl')
print("Pipeline (preprocessor + model) saved to model.pkl")