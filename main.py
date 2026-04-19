# ================================
# EXPENSE TRACKER DATA SCIENCE APP
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression

sns.set_style('whitegrid')

# -------------------------------
# 1. CREATE SYNTHETIC DATA
# -------------------------------

def generate_expense_data(num_records=300):
    np.random.seed(42)

    categories = ['Food', 'Rent', 'Travel', 'Shopping', 'Utilities', 'Entertainment']
    payment_methods = ['Cash', 'Credit Card', 'UPI', 'Debit Card']

    start_date = datetime(2024, 1, 1)
    data = []

    for _ in range(num_records):
        date = start_date + timedelta(days=np.random.randint(0, 180))
        category = np.random.choice(categories)
        amount = round(np.random.uniform(100, 5000), 2)
        payment = np.random.choice(payment_methods)
        data.append([date, category, amount, payment])

    df = pd.DataFrame(data, columns=['Date', 'Category', 'Amount', 'Payment_Method'])
    return df


# -------------------------------
# 2. CLEAN DATA
# -------------------------------

def clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df['Amount'] = df['Amount'].astype(float)
    return df


# -------------------------------
# 3. ANALYSIS FUNCTIONS
# -------------------------------

def category_analysis(df):
    return df.groupby('Category')['Amount'].sum().sort_values(ascending=False)


def monthly_trend(df):
    return df.groupby('Month')['Amount'].sum().sort_index()


def payment_analysis(df):
    return df.groupby('Payment_Method')['Amount'].sum().sort_values(ascending=False)


def detect_overspending(df):
    threshold = df['Amount'].mean() * 1.5
    over_df = df[df['Amount'] > threshold].copy()
    over_df = over_df.sort_values(by='Amount', ascending=False)
    return over_df, threshold


def predict_future_expenses(df, months=3):
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    monthly['Month'] = monthly['Month'].astype(str)
    monthly['Month_Index'] = range(len(monthly))

    X = monthly[['Month_Index']].to_numpy()
    y = monthly['Amount'].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    future_months = np.arange(len(monthly), len(monthly) + months).reshape(-1, 1)
    predictions = model.predict(future_months)

    last_month = pd.Period(monthly['Month'].iloc[-1])
    future_labels = [(last_month + i + 1).strftime('%Y-%m') for i in range(months)]

    pred_df = pd.DataFrame({
        'Month': future_labels,
        'Predicted_Expense': [round(x, 2) for x in predictions]
    })
    return monthly, pred_df


# -------------------------------
# 4. VISUALIZATION
# -------------------------------

def plot_category_spending(category_data):
    plt.figure(figsize=(8, 5))
    category_data.plot(kind='bar', color='#4C72B0')
    plt.title('Category-wise Spending')
    plt.xlabel('Category')
    plt.ylabel('Total Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/category_spending.png')
    plt.close()


def plot_monthly_trend(monthly_data):
    plt.figure(figsize=(8, 5))
    plt.plot(monthly_data.index.astype(str), monthly_data.values, marker='o', color='#55A868')
    plt.title('Monthly Spending Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/monthly_trend.png')
    plt.close()


def plot_payment_method(payment_data):
    plt.figure(figsize=(6, 6))
    payment_data.plot(kind='pie', autopct='%1.1f%%', startangle=140, cmap='tab20c')
    plt.title('Payment Method Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('outputs/payment_method.png')
    plt.close()


def plot_prediction(monthly_actual, predicted):
    plt.figure(figsize=(8, 5))
    plt.plot(monthly_actual['Month'].astype(str), monthly_actual['Amount'], marker='o', label='Actual', color='#4C72B0')
    plt.plot(predicted['Month'], predicted['Predicted_Expense'], marker='o', linestyle='--', label='Predicted', color='#DD8452')
    plt.title('Actual vs Predicted Expenses')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/expense_prediction.png')
    plt.close()


# -------------------------------
# 5. INSIGHTS GENERATION
# -------------------------------

def generate_insights(category_data, monthly_data, over_df, predicted):
    print('\n===== INSIGHTS =====')

    top_category = category_data.idxmax()
    print(f'👉 Highest spending category: {top_category}')

    highest_month = monthly_data.idxmax()
    print(f'👉 Highest spending month: {highest_month}')

    if top_category == 'Food':
        print('⚠️  Food is the top spending category. Review dining and delivery costs.')

    if monthly_data.max() > 150000:
        print('📈 Monthly spending spike detected. Investigate the high-cost month.')

    if len(over_df) > 10:
        print('🚨 Frequent overspending transactions detected.')

    print(f'👉 Predicted next month spend: ₹{predicted.loc[0, "Predicted_Expense"]}')
    print('👉 Recommendation: Use this analysis for budgeting and forecasting.')


# -------------------------------
# 6. MAIN PIPELINE
# -------------------------------

def main():
    print('Generating synthetic expense data...')
    df = generate_expense_data()

    print('Cleaning data...')
    df = clean_data(df)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/expenses.csv', index=False)

    print('Performing analysis...')
    category_data = category_analysis(df)
    monthly_data = monthly_trend(df)
    payment_data = payment_analysis(df)
    over_df, threshold = detect_overspending(df)
    monthly_actual, predicted = predict_future_expenses(df)

    os.makedirs('outputs', exist_ok=True)
    category_data.to_csv('outputs/category_analysis.csv')
    monthly_data.to_csv('outputs/monthly_trend.csv')
    payment_data.to_csv('outputs/payment_analysis.csv')
    over_df.to_csv('outputs/overspending_transactions.csv', index=False)
    predicted.to_csv('outputs/predicted_expenses.csv', index=False)

    print('\nTop Category Spending:\n', category_data)
    print('\nMonthly Trend:\n', monthly_data)
    print('\nOverspending Transactions (top 5):\n', over_df.head())
    print('\nExpense Prediction:\n', predicted)

    print('Generating visualizations...')
    plot_category_spending(category_data)
    plot_monthly_trend(monthly_data)
    plot_payment_method(payment_data)
    plot_prediction(monthly_actual, predicted)

    generate_insights(category_data, monthly_data, over_df, predicted)

    print('\nProject artifacts saved in data/ and outputs/')


if __name__ == '__main__':
    main()
