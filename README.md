# Expense Tracker App using Data Science

## Overview
This project is a beginner-friendly Expense Tracker App built with synthetic expense data and data science analysis.
It is designed for students preparing for internships and placements in Data Analyst, Business Analyst, and Financial Analyst roles.

## What it does
- Generates a synthetic expense dataset
- Cleans and preprocesses expense records
- Analyzes spending by category, month, and payment method
- Detects overspending transactions
- Predicts future expenses using a simple machine learning model
- Generates charts and saves results to `outputs/`

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit (optional dashboard)

## Project Structure
```
Expense Tracker App using Data Science/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── data/
│   └── expenses.csv
├── outputs/
│   ├── category_analysis.csv
│   ├── monthly_trend.csv
│   ├── payment_analysis.csv
│   ├── overspending_transactions.csv
│   ├── predicted_expenses.csv
│   ├── category_spending.png
│   ├── monthly_trend.png
│   ├── payment_method.png
│   └── expense_prediction.png
├── notebooks/
├── src/
└── images/
```

## Installation
1. Create and activate a virtual environment
   - Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Running the project
- Run console analysis:
  ```bash
  python main.py
  ```
- Run the Streamlit dashboard:
  ```bash
  streamlit run app.py
  ```

## Expected outputs
- `data/expenses.csv` - generated expense dataset
- `outputs/` - saved charts, analysis CSVs, and predictions
- `category_spending.png`, `monthly_trend.png`, `payment_method.png`, `expense_prediction.png`

## Why this project is valuable
- Demonstrates data generation, cleaning, and analysis skills
- Shows data visualization and reporting
- Includes a simple ML prediction model
- Supports a portfolio-ready GitHub project

## Notes
- The app uses synthetic data so it is easy to demonstrate without real financial records.
- The Streamlit dashboard provides an interactive UI for data exploration.
