import streamlit as st
import pandas as pd
import pickle


def load_data():
    with open('HR_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_data()

model = data["model"]
le_salary = data["le_salary"]

def show_predict_page():
    st.title('Exit Insight: A Machine Learning Predictor of Employee Retention')
    statisfaction = st.slider('Statisfaction Level', 0.09, 1.00, 0.50)
    hours = st.slider('Average Monthly Hours', 96, 310, 150)
    promotion = st.selectbox('Promotion Within Last 5 Years',(0,1))
    salary = st.selectbox('Salary Category', ('low', 'medium', 'high'))
    btnAction = st.button('Make Prediction')

    if btnAction:
        predictors = pd.DataFrame({
        "satisfaction_level":[statisfaction],
        "average_montly_hours" : [hours],
        "promotion_last_5years": [promotion],
        "salary": [salary]
        })
        predictors.salary = le_salary.transform(predictors.salary)

        result = model.predict(predictors)

        def prediction_():
            if result == 0:
                return 'stay.'
            else:
                return 'leave.'

        st.write(f"The employee might {prediction_()}")
