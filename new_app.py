import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("C:/Users/Admin/Desktop/Employee Salary Prediction App/salary.csv")

# Fetch live exchange rates (Replace with a real API key if needed)
def get_exchange_rates():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data["rates"]
    except:
        return {"USD": 1.0, "EUR": 0.91, "GBP": 0.78, "INR": 83.5, "AUD": 1.52, "CAD": 1.36}

currency_rates = get_exchange_rates()

# Convert all salaries to USD before training
df["salary_in_usd"] = df.apply(lambda row: row["salary"] / currency_rates.get(row["salary_currency"], 1), axis=1)

# Selecting relevant features
features = ['job_title', 'experience_level', 'employment_type', 'work_models',
            'employee_residence', 'company_location', 'company_size']
target = 'salary_in_usd'

# Splitting the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-Hot Encoding for categorical variables
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), features)])

# Using Gradient Boosting for better accuracy
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
])

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "salary_model.pkl")

# Streamlit App UI Enhancements
st.set_page_config(page_title="AI Salary Predictor", page_icon="💰", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: url('https://source.unsplash.com/1600x900/?finance,technology') no-repeat center center fixed; background-size: cover; color: white; }
    .main-title { text-align: center; font-size: 3rem; color: #FFD700; font-weight: bold; }
    .stButton>button { background: #4CAF50; color: white; font-size: 18px; padding: 10px; border-radius: 10px; }
    .result-card { background-color: #ffffff; color: #000; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.2); }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>💰 AI-Based Salary Prediction App</h1>", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=100)
st.sidebar.title("Enter Details")

job_title = st.sidebar.text_input("Job Title")
experience_level = st.sidebar.selectbox("Experience Level", ['Entry-level', 'Mid-level', 'Senior-level'])
employment_type = st.sidebar.selectbox("Employment Type", ['Full-time', 'Part-time', 'Contract', 'Freelance'])
work_models = st.sidebar.selectbox("Work Model", ['Remote', 'On-site', 'Hybrid'])
employee_residence = st.sidebar.text_input("Employee Residence (Country)")
company_location = st.sidebar.text_input("Company Location (Country)")
company_size = st.sidebar.selectbox("Company Size", ['Small', 'Medium', 'Large'])
salary_currency = st.sidebar.selectbox("Salary Currency", ['USD', 'EUR', 'GBP', 'INR', 'AUD', 'CAD'])

# Predict salary
if st.sidebar.button("Predict Salary"):
    with st.spinner("Calculating salary..."):
        try:
            # Load trained model
            model = joblib.load("salary_model.pkl")

            # Create DataFrame from user input
            input_data = pd.DataFrame([[job_title, experience_level, employment_type, work_models,
                                        employee_residence, company_location, company_size]],
                                      columns=features)
            
            # Predict salary in USD
            predicted_salary_usd = model.predict(input_data)[0]

            # Convert to selected currency
            converted_salary = predicted_salary_usd * currency_rates.get(salary_currency, 1)
            
            st.markdown(f"""
            <div class='result-card'>
                <h2>💵 Predicted Salary</h2>
                <h3>{round(converted_salary, 2)} {salary_currency}</h3>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error("Invalid input. Please check your entries.")

# Salary Distribution Chart
st.subheader("📊 Salary Distribution by Job Title")
fig = px.histogram(df, x="job_title", y="salary_in_usd", title="Salary Distribution by Job Title",
                   labels={"salary_in_usd": "Salary (USD)"}, color_discrete_sequence=["#4CAF50"])
st.plotly_chart(fig)

st.subheader("📈 Salary vs Experience Level")
fig_exp = px.box(df, x="experience_level", y="salary_in_usd", color="experience_level",
                 title="Salary Range by Experience Level")
st.plotly_chart(fig_exp)




