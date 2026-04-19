# =========================================
# STREAMLIT EXPENSE TRACKER DASHBOARD
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Expense Tracker", layout="wide")

# =========================================
# DATABASE SETUP
# =========================================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT,
                    password TEXT
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS expenses (
                    username TEXT,
                    date TEXT,
                    category TEXT,
                    amount REAL,
                    payment TEXT
                )''')

    conn.commit()
    conn.close()


def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # Check if user already exists
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    if c.fetchone():
        conn.close()
        return False
    
    c.execute("INSERT INTO users VALUES (?,?)", (username, password))
    conn.commit()
    conn.close()
    return True


def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    data = c.fetchone()
    conn.close()
    return data


def load_user_expenses(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT date, category, amount, payment FROM expenses WHERE username=?", (username,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=['Date', 'Category', 'Amount', 'Payment_Method', 'Month'])

    user_df = pd.DataFrame(rows, columns=['Date', 'Category', 'Amount', 'Payment_Method'])
    user_df['Date'] = pd.to_datetime(user_df['Date'])
    user_df['Month'] = user_df['Date'].dt.to_period('M')
    user_df['Amount'] = user_df['Amount'].astype(float)
    return user_df


# =========================================
# INITIALIZE DATABASE
# =========================================
init_db()

# =========================================
# LOGIN & REGISTRATION
# =========================================
st.sidebar.title("🔐 Login System")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type='password')

    if st.sidebar.button("Register"):
        if register_user(new_user, new_pass):
            st.sidebar.success("User registered! You can now login.")
        else:
            st.sidebar.error("User already exists!")

elif choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')

    if st.sidebar.button("Login"):
        result = login_user(username, password)
        if result:
            st.session_state['user'] = username
            st.success(f"Welcome {username}")
        else:
            st.error("Invalid credentials")

    if st.sidebar.button("Logout"):
        if 'user' in st.session_state:
            del st.session_state['user']
            st.rerun()

if 'user' not in st.session_state:
    st.warning("Please login to continue")
    st.stop()

# =========================================
# ML MODEL - EXPENSE PREDICTION
# =========================================

def predict_future_expenses(df):

    # Convert Month to numeric index
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    monthly['Month'] = monthly['Month'].astype(str)

    monthly['Month_Index'] = range(len(monthly))

    X = monthly[['Month_Index']]
    y = monthly['Amount']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 3 months
    future_months = np.array(range(len(monthly), len(monthly)+3)).reshape(-1,1)
    predictions = model.predict(future_months)

    # Create future labels
    last_month = pd.Period(monthly['Month'].iloc[-1])
    future_labels = [(last_month + i + 1).strftime('%Y-%m') for i in range(3)]

    pred_df = pd.DataFrame({
        'Month': future_labels,
        'Predicted_Expense': [round(x, 2) for x in predictions]
    })

    return monthly, pred_df

# =========================================
# ARIMA FORECASTING
# =========================================
def arima_forecast(df):
    monthly = df.groupby('Month')['Amount'].sum()
    monthly.index = monthly.index.to_timestamp()

    model = ARIMA(monthly, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=3)

    forecast_df = forecast.reset_index()
    forecast_df.columns = ['Month', 'Forecast']
    forecast_df['Month'] = forecast_df['Month'].dt.strftime('%Y-%m')

    return monthly, forecast_df


# =========================================
# PDF REPORT GENERATOR
# =========================================
def generate_pdf(total, avg, max_val):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Expense Report", styles['Title']))
    content.append(Paragraph(f"Total Spending: ₹{round(total,2)}", styles['Normal']))
    content.append(Paragraph(f"Average Spending: ₹{round(avg,2)}", styles['Normal']))
    content.append(Paragraph(f"Highest Expense: ₹{round(max_val,2)}", styles['Normal']))

    doc.build(content)


# -------------------------------
# 1. SYNTHETIC DATA GENERATION
# -------------------------------
def generate_data(n=300):
    np.random.seed(42)

    categories = ['Food', 'Rent', 'Travel', 'Shopping', 'Utilities', 'Entertainment']
    payment_methods = ['Cash', 'UPI', 'Credit Card', 'Debit Card']

    start_date = datetime(2024, 1, 1)
    data = []

    for _ in range(n):
        date = start_date + timedelta(days=np.random.randint(0, 180))
        category = np.random.choice(categories)
        amount = round(np.random.uniform(100, 5000), 2)
        payment = np.random.choice(payment_methods)

        data.append([date, category, amount, payment])

    df = pd.DataFrame(data, columns=['Date', 'Category', 'Amount', 'Payment_Method'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    return df


# -------------------------------
# 2. LOAD DATA
# -------------------------------
st.sidebar.title("⚙️ Options")

data_option = st.sidebar.radio(
    "Select Data Source:",
    ["Use Synthetic Data", "Upload CSV"]
)

if data_option == "Upload CSV":
    file = st.sidebar.file_uploader("Upload your expense CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
    else:
        st.warning("Please upload a CSV file")
        st.stop()
else:
    df = generate_data()

# Load user-specific expenses
user_expenses = load_user_expenses(st.session_state['user'])
if not user_expenses.empty:
    df = user_expenses
else:
    df = generate_data()

# -------------------------------
# USER EXPENSE INPUT
# -------------------------------
st.subheader("➕ Add Expense")

expense_date = st.date_input("Date", datetime.today())
expense_category = st.selectbox("Category", ['Food', 'Rent', 'Travel', 'Shopping', 'Utilities', 'Entertainment'])
expense_amount = st.number_input("Amount", min_value=0.0, value=0.0, step=1.0)
expense_payment = st.selectbox("Payment Method", ['Cash', 'UPI', 'Credit Card', 'Debit Card'])

if st.button("Add Expense"):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO expenses VALUES (?,?,?,?,?)", (st.session_state['user'], str(expense_date), expense_category, expense_amount, expense_payment))
    conn.commit()
    conn.close()
    st.success("Expense added!")
    st.rerun()

# -------------------------------
# 3. FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

df = df[df['Category'].isin(category_filter)]

# -------------------------------
# 4. KPI METRICS
# -------------------------------
total_spent = df['Amount'].sum()
avg_spent = df['Amount'].mean()
max_spent = df['Amount'].max()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Spending", f"₹ {round(total_spent,2)}")
col2.metric("📊 Avg Transaction", f"₹ {round(avg_spent,2)}")
col3.metric("🔥 Highest Expense", f"₹ {round(max_spent,2)}")

if st.button("📄 Download Report"):
    generate_pdf(total_spent, avg_spent, max_spent)
    with open("report.pdf", "rb") as f:
        st.download_button("Download PDF", f, file_name="expense_report.pdf")

st.markdown("---")

# -------------------------------
# 5. CATEGORY ANALYSIS
# -------------------------------
st.subheader("📊 Category-wise Spending")

category_data = df.groupby('Category')['Amount'].sum()

fig1, ax1 = plt.subplots()
category_data.plot(kind='bar', ax=ax1)
ax1.set_title("Category Spending")
st.pyplot(fig1)

# -------------------------------
# 6. MONTHLY TREND
# -------------------------------
st.subheader("📈 Monthly Spending Trend")

monthly_data = df.groupby('Month')['Amount'].sum()

fig2, ax2 = plt.subplots()
monthly_data.plot(marker='o', ax=ax2)
ax2.set_title("Monthly Trend")
st.pyplot(fig2)

# -------------------------------
# 7. PAYMENT METHOD
# -------------------------------
st.subheader("💳 Payment Distribution")

payment_data = df.groupby('Payment_Method')['Amount'].sum()

fig3, ax3 = plt.subplots()
payment_data.plot(kind='pie', autopct='%1.1f%%', ax=ax3)
ax3.set_ylabel("")
st.pyplot(fig3)

# -------------------------------
# 8. OVESPENDING DETECTION
# -------------------------------
st.subheader("⚠️ Overspending Transactions")

threshold = df['Amount'].mean() * 1.5
over_df = df[df['Amount'] > threshold]

st.dataframe(over_df.head())

# -------------------------------
# 9. BUDGET ANALYSIS
# -------------------------------
st.subheader("💸 Budget Analysis")

budget = st.slider("Select Monthly Budget", 10000, 100000, 50000)

monthly_sum = df.groupby('Month')['Amount'].sum()

for month, value in monthly_sum.items():
    if value > budget:
        st.error(f"{month}: ₹{round(value,2)} (Exceeded)")
    else:
        st.success(f"{month}: ₹{round(value,2)} (Within Budget)")

# -------------------------------
# 10. INSIGHTS
# -------------------------------
st.subheader("🧠 Insights")

top_category = category_data.idxmax()
top_month = monthly_data.idxmax()

st.write(f"👉 Highest spending category: **{top_category}**")
st.write(f"👉 Highest spending month: **{top_month}**")
st.write("👉 Recommendation: Reduce spending in high-expense categories")

st.subheader("📌 Advanced Insights")

if category_data.idxmax() == "Food":
    st.warning("🍔 High spending on Food detected. Consider reducing online orders.")

if monthly_data.max() > 150000:
    st.warning("📈 Sudden spike in monthly spending. Investigate unusual expenses.")

if len(over_df) > 10:
    st.error("⚠️ Frequent overspending transactions detected!")

# -------------------------------
# ML PREDICTION SECTION
# -------------------------------

st.subheader("🔮 Future Expense Prediction (ML)")

monthly_actual, predicted = predict_future_expenses(df)

st.write("### Predicted Expenses for Next 3 Months")
st.dataframe(predicted)

# Plot
fig4, ax4 = plt.subplots()

# Actual
ax4.plot(monthly_actual['Month'], monthly_actual['Amount'], marker='o', label='Actual')

# Predicted
ax4.plot(predicted['Month'], predicted['Predicted_Expense'], marker='o', linestyle='--', label='Predicted')

ax4.set_title("Actual vs Predicted Expenses")
ax4.legend()

plt.xticks(rotation=45)

st.pyplot(fig4)

# -------------------------------
# ARIMA FORECASTING
# -------------------------------
st.subheader("📈 ARIMA Forecasting (Advanced ML)")

actual, forecast = arima_forecast(df)

st.dataframe(forecast.style.format({"Forecast": "₹ {:.2f}"}))

fig_arima, ax = plt.subplots()
ax.plot(actual.index.strftime('%Y-%m'), actual.values, marker='o', label='Actual')
ax.plot(forecast['Month'], forecast['Forecast'], marker='o', linestyle='--', label='ARIMA Forecast')
ax.set_title("ARIMA Forecast vs Actual")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig_arima)

# -------------------------------
# 11. RAW DATA
# -------------------------------
st.subheader("📂 Raw Data")
st.dataframe(df.head())