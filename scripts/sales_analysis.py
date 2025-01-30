import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the dataset path is correct
dataset_path = os.path.abspath('data/sales_data.csv')

# Load the dataset with proper encoding
try:
    data = pd.read_csv(dataset_path, encoding='ISO-8859-1')  # Fix for UnicodeDecodeError
    print("✅ Dataset Loaded Successfully!\n")
except FileNotFoundError:
    print("❌ Error: sales_data.csv not found in data/ folder.")
    exit()
except UnicodeDecodeError:
    print("❌ Encoding Error: Trying different encoding...")
    data = pd.read_csv(dataset_path, encoding='latin1')

# Display first few rows
print("📊 First 5 rows of dataset:")
print(data.head())

# Drop missing values
data.dropna(inplace=True)

# Create a new column for total sales
data['Total Sales'] = data['Sales'] * data['Quantity']

# Ensure images folder exists
if not os.path.exists('images'):
    os.makedirs('images')

# ---------------------------
# 🔹 Total Sales by Region
# ---------------------------
sales_by_region = data.groupby('Region')['Total Sales'].sum().sort_values(ascending=False)

print("\n📌 Total Sales by Region:")
print(sales_by_region)

# Create visualization
plt.figure(figsize=(8, 5))
sales_by_region.plot(kind='bar', color='royalblue')
plt.title('Total Sales by Region')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.savefig('images/sales_by_region.png')
plt.show()

# ---------------------------
# 🔹 Top 5 Best-Selling Products
# ---------------------------
top_products = data.groupby('Product Name')['Total Sales'].sum().sort_values(ascending=False).head(5)

print("\n📌 Top 5 Products by Sales:")
print(top_products)

# Create visualization
plt.figure(figsize=(8, 5))
top_products.plot(kind='bar', color='green')
plt.title('Top 5 Best-Selling Products')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.savefig('images/top_products.png')
plt.show()

# ---------------------------
# 🔹 Monthly Sales Trends
# ---------------------------
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data.dropna(subset=['Order Date'], inplace=True)  # Drop invalid dates
data['Month'] = data['Order Date'].dt.to_period('M')

monthly_sales = data.groupby('Month')['Total Sales'].sum()

print("\n📌 Monthly Sales Trends:")
print(monthly_sales)

# Create visualization
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o', color='red')
plt.title('Monthly Sales Trends')
plt.ylabel('Total Sales')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.savefig('images/monthly_sales.png')
plt.show()

print("\n✅ Analysis Completed! All charts saved in images/ folder.")
