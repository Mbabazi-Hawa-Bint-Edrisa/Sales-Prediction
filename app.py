from flask import Flask, render_template, request
import pandas as pd
import pickle
import sqlite3
from datetime import datetime
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the trained model and feature columns
try:
    with open('sales_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Model or feature columns file not found. Please ensure 'sales_model.pkl' and 'feature_columns.pkl' exist.")

# Load the maximum quantity sold per product per month for predictions
try:
    max_quantity_per_product_per_month = pd.read_csv('max_quantity_per_product_per_month.csv')
except FileNotFoundError:
    raise FileNotFoundError("max_quantity_per_product_per_month.csv not found.")

# Load the training dataset to calculate totals and lagged sales
try:
    train_df = pd.read_csv('notebook/cleaned_train_dataset.csv')
except FileNotFoundError:
    raise FileNotFoundError("cleaned_train_dataset.csv not found in notebook directory.")

# Function to split product_name into actual product name and rate
def split_product_and_rate(product_name):
    match = re.search(r'^(.*?)\s*(\d+)$', product_name.strip())
    if match:
        return match.group(1).strip(), float(match.group(2))
    return product_name.strip(), 0.0  # Default to 0.0 if no rate found

# Clean product names in the training dataset
train_df[['product_name_clean', 'rate_from_product']] = pd.DataFrame(
    train_df['product_name'].apply(split_product_and_rate).tolist(),
    index=train_df.index
)
train_df['product_name'] = train_df['product_name_clean']
train_df = train_df.drop(columns=['product_name_clean'])
train_df['rate_from_product'] = train_df['rate_from_product'].fillna(0.0).astype(float)
if 'rate' in train_df.columns:
    train_df['rate'] = train_df['rate'].fillna(0.0).astype(float)
    train_df['rate'] = train_df['rate'].combine_first(train_df['rate_from_product'])
else:
    train_df['rate'] = train_df['rate_from_product']
train_df = train_df.drop(columns=['rate_from_product'])

# Define the order of months for plotting and lagged sales
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
train_df['month'] = pd.Categorical(train_df['month'], categories=month_order, ordered=True)
train_df['month_idx'] = train_df['month'].map(lambda x: month_order.index(x))
train_df = train_df.sort_values(['product_name', 'month_idx'])

# Calculate total sales amount per product per month for the performance graph
product_performance = train_df.groupby(['product_name', 'month'], observed=True).agg(
    total_amount=pd.NamedAgg(column='amount', aggfunc='sum')
).reset_index()
product_performance = product_performance.sort_values(['product_name', 'month'])

# Database setup
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  product_name TEXT, unit_name TEXT, month TEXT,
                  rate REAL, prediction REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    plot_path = None  # Default value for GET requests
    error_message = None  # To store any error messages

    if request.method == 'POST':
        try:
            # Get form data
            product_name = request.form['product_name'].strip()
            unit_name = request.form['unit_name'].strip()
            month = request.form['month'].strip()
            rate = request.form['rate'].strip()

            # Validate inputs
            if not product_name or not unit_name or not month or not rate:
                raise ValueError("All fields (product name, unit name, month, rate) must be provided.")

            if month not in month_order:
                raise ValueError(f"Invalid month: {month}. Please select a valid month (e.g., January, February, etc.).")

            try:
                rate = float(rate)
                if rate <= 0:
                    raise ValueError("Rate must be a positive number.")
            except ValueError:
                raise ValueError("Rate must be a valid number.")

            # Keep the original product_name and unit_name for display
            display_product_name = product_name
            display_unit_name = unit_name

            # Clean the input product_name
            product_name_clean, _ = split_product_and_rate(product_name)
            product_name = product_name_clean

            # Filter product performance for the specific product
            product_data = product_performance[product_performance['product_name'] == product_name]

            # Pivot the data for plotting
            if not product_data.empty:
                pivot_df = product_data.pivot(index='month', columns='product_name', values='total_amount').fillna(0)

                # Create a line plot for the specific product's performance over months
                plt.figure(figsize=(12, 6))
                plt.plot(pivot_df.index, pivot_df[product_name], marker='o', label=product_name, color='blue')
                plt.title(f'Performance of {product_name} Over Months (Total Sales Amount)')
                plt.xlabel('Month')
                plt.ylabel('Total Sales Amount')
                plt.legend(title='Product')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

                # Ensure static directory exists
                if not os.path.exists('static'):
                    os.makedirs('static')

                # Save the plot to the static folder
                plot_filename = f'product_performance_{product_name.lower().replace(" ", "_")}.png'
                plot_path = os.path.join('static', plot_filename)
                plt.savefig(plot_path)
                plt.close()
            else:
                # If no data exists for the product, create a placeholder plot
                plt.figure(figsize=(12, 6))
                plt.plot(month_order, [0] * len(month_order), marker='o', label=product_name, color='blue')
                plt.title(f'Performance of {product_name} Over Months (No Data Available)')
                plt.xlabel('Month')
                plt.ylabel('Total Sales Amount')
                plt.legend(title='Product')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
                if not os.path.exists('static'):
                    os.makedirs('static')
                plot_filename = f'product_performance_{product_name.lower().replace(" ", "_")}.png'
                plot_path = os.path.join('static', plot_filename)
                plt.savefig(plot_path)
                plt.close()

            # Feature engineering: Extract month number
            month_number = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }[month]

            # Look up the maximum quantity for this product and month
            max_quantity_row = max_quantity_per_product_per_month[
                (max_quantity_per_product_per_month['product_name'] == product_name) &
                (max_quantity_per_product_per_month['month'] == month)
            ]
            if not max_quantity_row.empty:
                quantity = max_quantity_row['max_quantity'].iloc[0]
            else:
                # Fallback: Use the median max_quantity if the product/month combo isn't found
                quantity = max_quantity_per_product_per_month['max_quantity'].median()
                error_message = f"Note: Product '{product_name}' or month '{month}' not found in quantity data. Using median quantity for prediction."

            # Calculate lagged sales: Get the sales amount for the previous month from train_df
            current_month_idx = month_order.index(month)
            previous_month_idx = (current_month_idx - 1) % 12  # Wrap around for January
            previous_month = month_order[previous_month_idx]
            product_sales = train_df[train_df['product_name'] == product_name]
            previous_sales = product_sales[product_sales['month'] == previous_month]['amount']
            lagged_sales = previous_sales.iloc[-1] if not previous_sales.empty else 0

            # Create input DataFrame for the model
            input_data = pd.DataFrame({
                'product_name': [product_name],
                'unit_name': [unit_name],
                'month': [month],
                'rate': [rate],
                'quantity': [quantity],
                'month_number': [month_number],
                'rate_quantity_interaction': [rate * quantity],
                'quarter': [(month_number-1)//3 + 1],
                'lagged_sales': [lagged_sales]
            })

            # One-hot encode the categorical features
            categorical_features = ['product_name', 'unit_name', 'month']
            input_encoded = pd.get_dummies(input_data[categorical_features], columns=categorical_features, drop_first=True)

            # Combine with numerical features
            input_combined = pd.concat([input_encoded, input_data[['rate', 'quantity', 'month_number', 'rate_quantity_interaction', 'quarter', 'lagged_sales']]], axis=1)

            # Align columns with training data
            input_aligned = pd.DataFrame(0, index=[0], columns=feature_columns)
            for col in input_combined.columns:
                if col in input_aligned.columns:
                    input_aligned[col] = input_combined[col]

            # Make prediction using the model directly
            prediction = model.predict(input_aligned)[0]
            prediction = round(prediction, 2)  # Round to 2 decimal places

            # Save prediction to database
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            c.execute("INSERT INTO predictions (product_name, unit_name, month, rate, prediction, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                      (display_product_name, display_unit_name, month, rate, prediction, timestamp))
            conn.commit()
            conn.close()

            # Fetch prediction history
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            c.execute("SELECT product_name, unit_name, month, rate, prediction, timestamp FROM predictions ORDER BY timestamp DESC")
            history = c.fetchall()
            conn.close()

            return render_template('predict.html', prediction=prediction,
                                  product_name=display_product_name,
                                  unit_name=display_unit_name,
                                  month=month, rate=rate,
                                  history=history, plot_path=plot_path,
                                  error_message=error_message)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            # Fetch prediction history for display even if prediction fails
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            c.execute("SELECT product_name, unit_name, month, rate, prediction, timestamp FROM predictions ORDER BY timestamp DESC")
            history = c.fetchall()
            conn.close()
            return render_template('predict.html', history=history, plot_path=plot_path, error_message=error_message)

    # Fetch prediction history for GET request
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT product_name, unit_name, month, rate, prediction, timestamp FROM predictions ORDER BY timestamp DESC")
    history = c.fetchall()
    conn.close()

    return render_template('predict.html', history=history, plot_path=plot_path, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)