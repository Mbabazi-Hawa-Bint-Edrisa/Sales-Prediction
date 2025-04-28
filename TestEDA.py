import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the test dataset
df = pd.read_csv('Test.csv')

# --- Data Cleaning ---
# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Standardize the month column (keep as string, just clean up formatting)
df['month'] = df['month'].astype(str).str.strip().str.title()

# Add month_number for consistency with training
df['month_number'] = df['month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
})

# Standardize product_code and product_name mapping
code_name_mapping = df.groupby('product_code')['product_name'].nunique()
inconsistent_codes = code_name_mapping[code_name_mapping > 1]
for code in inconsistent_codes.index:
    most_frequent_name = df[df['product_code'] == code]['product_name'].mode()[0]
    df.loc[df['product_code'] == code, 'product_name'] = most_frequent_name

# Trim spaces and standardize strings
df['product_name'] = df['product_name'].str.strip()
df['unit_name'] = df['unit_name'].str.strip().str.title()

# Limit categories for product_name and unit_name (same as train_model.py)
for feature in ['product_name', 'unit_name']:
    if feature == 'product_name':
        top_categories = df[feature].value_counts().head(50).index
        df[feature] = df[feature].apply(lambda x: x if x in top_categories else 'Other')
    elif feature == 'unit_name':
        top_categories = df[feature].value_counts().head(20).index
        df[feature] = df[feature].apply(lambda x: x if x in top_categories else 'Other')

# Cap outliers for rate and amount (using bounds from TrainEDA.py output if available)
# For now, we'll calculate bounds locally
for col in ['rate', 'amount']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound).astype(int)

# Save the cleaned dataset
df.to_csv('cleaned_test_dataset.csv', index=False)
print("Cleaned test dataset saved as 'cleaned_test_dataset.csv'")

# --- Data Assessment ---
print("\n### 1. Data Assessment")
print("\nShape of the dataset:", df.shape)
print("\nColumns in the dataset:", df.columns)
print("\nData types and info:")
df.info()
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nNumber of duplicated rows:", df.duplicated().sum())

# Define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print('\nWe have {} numerical features in test data: {}'.format(len(numeric_features), numeric_features))
print('We have {} categorical features in test data: {}'.format(len(categorical_features), categorical_features))

# Check unique values in categorical features
print('\nNumber of unique data points in categorical features in the test dataset:')
print('Number of unique data points in Product code:', df['product_code'].nunique(), df['product_code'].unique())
print('Number of unique data points in Product name:', df['product_name'].nunique(), df['product_name'].unique())
print('Number of unique data points in Unit name:', df['unit_name'].nunique(), df['unit_name'].unique())
print('Number of unique data points in Month:', df['month'].nunique(), df['month'].unique())

# --- Additional Cleaning Checks ---
# Verify one-to-one mapping between 'product_code' and 'product_name'
code_name_mapping = df.groupby('product_code')['product_name'].nunique()
inconsistent_codes = code_name_mapping[code_name_mapping > 1]
if not inconsistent_codes.empty:
    print("\nProduct codes with multiple names after fix:")
    print(inconsistent_codes)
else:
    print("\nAll product codes map to exactly one product name after fix.")

# Check for case inconsistencies or extra spaces in 'unit_name' and 'month'
print("\nUnique values in 'unit_name':", df['unit_name'].unique())
print("Unique values in 'month':", df['month'].unique())

# Validate numeric columns: Check for negative values
negative_rate = df[df['rate'] < 0]
if not negative_rate.empty:
    print("\nRows with negative rate:")
    print(negative_rate[['product_code', 'product_name', 'rate']])

# --- Product Performance Analysis ---
print("\n### Product Performance")
sns.set(style="whitegrid")
product_performance = df.groupby('product_name').agg({
    'amount': 'sum',
    'tax': 'sum',
    'product_code': 'first'
}).reset_index()
top_products = product_performance.sort_values(by='amount', ascending=False).head(10)
print("\nTop 10 Products by Total Amount:")
print(top_products[['product_name', 'product_code', 'amount']])

# --- Univariate EDA ---
print("\n### Univariate EDA")
numeric_cols = ['rate', 'amount', 'tax', 'month_number']
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 4, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('test_distribution_plots.png')
plt.close()

categorical_cols = ['unit_name', 'month']
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 2, i)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('test_count_plots.png')
plt.close()

# --- Bivariate EDA ---
print("\n### Bivariate EDA")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_performance, x='amount', y='tax', hue='amount', size='amount')
plt.title('Scatter Plot: Amount vs. Tax by Product')
plt.xlabel('Total Amount')
plt.ylabel('Total Tax')
plt.savefig('test_scatter_amount_vs_tax.png')
plt.close()

# --- Multivariate EDA ---
print("\n### Multivariate EDA")
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Columns')
plt.savefig('test_correlation_matrix.png')
plt.close()

vif_data = df[numeric_cols].dropna()
vif_results = pd.DataFrame()
vif_results['Feature'] = numeric_cols
vif_results['VIF'] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
print("\nVariance Inflation Factor (VIF) for Numeric Columns:")
print(vif_results)

# --- Outlier Detection ---
print("\n### Outlier Detection")
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.savefig('test_box_plots.png')
plt.close()

outliers_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_summary[col] = {
        'num_outliers': len(outliers),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
print("\nOutlier Summary (IQR Method):")
for col, info in outliers_summary.items():
    print(f"{col}: {info['num_outliers']} outliers (lower: {info['lower_bound']:.2f}, upper: {info['upper_bound']:.2f})")

gross_outliers = product_performance[
    (product_performance['amount'] < outliers_summary['amount']['lower_bound']) |
    (product_performance['amount'] > outliers_summary['amount']['upper_bound'])
]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_performance, x='amount', y='tax', color='blue', label='Normal')
sns.scatterplot(data=gross_outliers, x='amount', y='tax', color='red', label='Outliers')
plt.title('Scatter Plot with Outliers in Amount')
plt.xlabel('Total Amount')
plt.ylabel('Total Tax')
plt.legend()
plt.savefig('test_outliers_amount.png')
plt.close()

# --- Save Top Products for Stocking Recommendations ---
top_products.to_csv('test_top_products.csv', index=False)
print("\nTop products saved to 'test_top_products.csv' for stocking recommendations.")