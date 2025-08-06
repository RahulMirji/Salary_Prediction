import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd
import os
# Title and Divider
st.set_page_config(page_title="Salary Estimation App", layout='wide')

st.markdown("""
    <h1 style='text-align: center;'>Salary Estimation App</h1>
    <p style='text-align: center;'>Predict your expected salary based on company experience!</p>
""", unsafe_allow_html=True)


# Image (fixed typo in use_container_width)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(r"https://tse2.mm.bing.net/th/id/OIP.tTeaImkFfg4t2ia1yvbFwgHaEK?r=0&rs=1&pid=ImgDetMain&o=7&rm=3", caption="Let's predict", use_container_width=True)


# Divider
st.divider()

# Inputs
col1, col2, col3 = st.columns(3)

with col1:
    years_at_company = st.number_input("Years at company", min_value=0, max_value=20, value=3)  # fixed typo: vlaue -> value

with col2:
    satisfaction_level = st.slider("Satisfaction level", min_value=0.0, max_value=1.0, step=0.01, value=0.7)

with col3:
    average_monthly_hours = st.slider("Avg Monthly Hours", min_value=120, max_value=310, step=1, value=160)  # fixed typo: vlaue -> value

# Collect input features into a list
X = [years_at_company, satisfaction_level, average_monthly_hours]

# Load model and scaler
base_path = os.path.dirname(__file__)  # Gets current script directory
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
model = joblib.load(os.path.join(base_path, "model.pkl"))
# Predict button
predict_button = st.button("Predict Salary")

st.divider()

if predict_button:
    st.balloons()

    # Prepare data for prediction
    X_array = scaler.transform([np.array(X)])  # fixed error: np.array[X] -> np.array(X)
    prediction = model.predict(X_array)

    # Display the prediction
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")

    # Visualize user input
    df_input = pd.DataFrame({
        "Feature": ["years_at_company", "satisfaction_level", "average_monthly_hours"],
        "Value": X
    })

    fig = px.bar(df_input, x="Feature", y="Value", color="Feature", title="Your Input Profile", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please enter your details and click Predict Salary to see the estimated salary.")
