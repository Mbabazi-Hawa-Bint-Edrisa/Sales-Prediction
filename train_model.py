import pandas as pd
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Start timing the script
start_time = time.time()
print("Starting script...")

# Load the cleaned training dataset
print("Loading dataset...")
df = pd.read_csv(r'C:\Users\USER\Desktop\sales pred\notebook\cleaned_train_dataset.csv')
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds. Shape: {df.shape}")

# Define features and target
features = ['product_name', 'unit_name', 'month', 'rate', 'quantity']  # Added rate and quantity
target = 'amount'

# Feature engineering: Extract month number for seasonality
df['month_number'] = df['month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
})
features.append('month_number')

# Limit categories for product_name and unit_name
print("Encoding features...")
encode_start = time.time()
for feature in ['product_name', 'unit_name']:
    if feature == 'product_name':
        top_categories = df[feature].value_counts().head(50).index
        df[feature] = df[feature].apply(lambda x: x if x in top_categories else 'Other')
    elif feature == 'unit_name':
        top_categories = df[feature].value_counts().head(20).index
        df[feature] = df[feature].apply(lambda x: x if x in top_categories else 'Other')

# One-hot encode categorical features
categorical_features = ['product_name', 'unit_name', 'month']
df_encoded = pd.get_dummies(df[categorical_features], columns=categorical_features, drop_first=True)

# Combine encoded categorical features with numerical features
X = pd.concat([df_encoded, df[['rate', 'quantity', 'month_number']]], axis=1)
y = df[target]
print(f"Encoding completed in {time.time() - encode_start:.2f} seconds. New shape: {X.shape}")

# Split the data into training and validation sets
print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model with tuned parameters
print("Training model...")
train_start = time.time()
model = RandomForestRegressor(
    n_estimators=100,  # Increased back to 100
    max_depth=10,  # Limit tree depth to prevent overfitting
    min_samples_split=5,  # Require at least 5 samples to split
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print(f"Model training completed in {time.time() - train_start:.2f} seconds.")

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Model Performance on Validation Set:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Save the trained model
with open('sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Trained model saved as 'sales_model.pkl'")

# Save the feature columns (for one-hot encoding during prediction)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Feature columns saved as 'feature_columns.pkl'")

# Total runtime
print(f"Total runtime: {time.time() - start_time:.2f} seconds.")