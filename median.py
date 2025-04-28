import pandas as pd

# Load the training dataset
try:
    df = pd.read_csv(r'C:\Users\USER\Desktop\sales pred\notebook\cleaned_train_dataset.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns in dataset: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Check unique product names to see if '10LTR JERRICAN 6000' exists
print("\nUnique product names in dataset:")
unique_products = df['product_name'].unique()
print(unique_products)
if '10LTR JERRICAN 6000' in unique_products:
    print("Found '10LTR JERRICAN 6000' in the dataset!")
else:
    print("Did NOT find '10LTR JERRICAN 6000' in the dataset!")

# Ensure 'month' is in a consistent format
df['month'] = df['month'].str.strip().str.title()

# Calculate the maximum quantity sold per product per month
max_quantity_per_product_per_month = df.groupby(['product_name', 'month'])['quantity'].max().reset_index()
max_quantity_per_product_per_month = max_quantity_per_product_per_month.rename(columns={'quantity': 'max_quantity'})

# Print the first 10 rows of the grouped data
print("\nFirst 10 rows after grouping:")
print(max_quantity_per_product_per_month.head(10))

# Save the results to a CSV for reference
max_quantity_per_product_per_month.to_csv('max_quantity_per_product_per_month.csv', index=False)
print("\nMaximum quantity sold per product per month saved to 'max_quantity_per_product_per_month.csv'")

# Filter for a specific product to double-check
product_to_check = '10LTR JERRICAN 6000'  # Updated to match the exact product name
product_data = max_quantity_per_product_per_month[max_quantity_per_product_per_month['product_name'] == product_to_check]
print(f"\nMaximum quantity for {product_to_check} by month:")
print(product_data)

# Additional check: How many rows for this product in the original dataset?
product_in_original = df[df['product_name'] == product_to_check]
print(f"\nNumber of rows for {product_to_check} in original dataset: {len(product_in_original)}")
if len(product_in_original) > 0:
    print(f"Sample rows for {product_to_check}:")
    print(product_in_original.head())