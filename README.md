# 📊 Superstore Sales Analysis — Advanced Dashboard  

https://superstore-salesanalysis.streamlit.app/

An **interactive, end-to-end business intelligence dashboard** built with **Streamlit, Plotly, Prophet, and Machine Learning techniques**.  
This project simulates a **real-world retail analytics system** that a data analyst or business intelligence engineer would deliver to stakeholders.  

---

## 🚀 Features  

✅ **Dynamic Filters** — Date, Region, Category, Segment, Ship Mode  
✅ **KPIs** — Sales, Profit, Orders, Profit Margin (with YoY delta)  
✅ **Time Series Forecasting** — Prophet-based forecasting (6+ months) with safe fallback method  
✅ **Customer RFM Analysis** — Recency, Frequency, Monetary segmentation + cohort view  
✅ **Product Leaderboard & ABC Analysis** — Identify A/B/C class products  
✅ **Market Basket Analysis (Association Rules)** — Understand product co-purchase patterns  
✅ **Geo Analysis** — U.S. sales choropleth by State (auto-detects dataset geography)  
✅ **Export Reports** — Download Excel (multi-sheet) and PDF business summary  
✅ **Interactive Visuals** — Built with Plotly for high interactivity  
✅ **Scalable Design** — Works with Kaggle / retail datasets (not just Superstore)  

---

## 📂 Project Structure  

```bash
Superstore_Sales_Analysis/
│
├── app.py                 # Streamlit app (main entry)
├── data/                  # Place your dataset(s) here
│   └── sales_data.csv     # Default dataset (replaceable)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## ⚡ Installation & Setup  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/VIGNESH54/Superstore_Sales_Analysis.git
cd Superstore_Sales_Analysis
```

### 2️⃣ Create & activate virtual environment  
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app  
```bash
streamlit run app.py
```

---

## 📊 Dataset  

- The app expects a dataset similar to the **Superstore dataset** (widely available on Kaggle).  
- Required columns:  
  ```
  Order Date, Order ID, Product Name, Sales, Profit, Customer ID
  ```
- Optional columns for advanced features:  
  ```
  Discount, Quantity, Region, Category, Sub-Category, Segment, Ship Mode, State
  ```

You can replace the default dataset with your own retail sales data.

---

## 📸 Screenshots  

### 🔹 Dashboard Overview  
*(insert screenshot here)*  

### 🔹 Forecasting (Prophet)  
*(insert screenshot here)*  

### 🔹 Geo Analysis  
*(insert screenshot here)*  

---

## 🧑‍💻 Tech Stack  

- **Frontend / UI**: Streamlit, Plotly  
- **Data Analysis**: Pandas, NumPy, Seaborn  
- **Forecasting**: Prophet (fallback to rolling mean)  
- **ML / Association Rules**: mlxtend (Apriori, Market Basket Analysis)  
- **Reports**: Matplotlib (PDF export), OpenPyXL (Excel export)  

---

## 🎯 Why This Project?  

This project demonstrates **professional-grade analytics skills**:  
- Data cleaning & preprocessing  
- Business KPI computation  
- Advanced analytics (RFM, forecasting, association rules)  
- Interactive dashboards for stakeholders  
- Report automation  

Perfect for **Data Analyst / BI / Data Science portfolios**.  

---

## 📦 Deployment  

Deployed on **Streamlit Cloud**:  
👉 https://superstore-salesanalysis.streamlit.app/

---

## 👤 Author  

**Vignesh P**  
📍 B.Tech CSE, SRM Institute of Science and Technology   
