import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.datasets import load_diabetes

# Load the trained model
model_path = 'model_ridge.pkl'  # Update this path if necessary
with open(model_path, 'rb') as file:
    model = pickle.load(file)

diab = load_diabetes()
X = pd.DataFrame(diab.data,columns=diab.feature_names)

user_input = {}

for col in X.columns:
    user_input[col]=st.slider(col, X[col].min(), X[col].max())

df = pd.DataFrame(user_input,index=[0])
st.write(df)
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(df)[0]

    # Display the prediction
    st.success(f"The predicted value is: {prediction:.2f}")

# # Define the Streamlit UI
# def main():
#     st.title("Regression Model Prediction")

#     st.write("This application uses a trained regression model to make predictions based on user inputs.")

#     # Add input fields for the features
#     st.header("Input Features")
    
    
#     age = st.number_input("Age", min_value=0, max_value=120, value=30)
#     sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
#     bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
#     bp = st.number_input("BP", min_value=0.0, max_value=200.0, value=80.0)
#     s1 = st.number_input("S1", min_value=0.0, max_value=500.0, value=100.0)
#     s2 = st.number_input("S2", min_value=0.0, max_value=500.0, value=100.0)
#     s3 = st.number_input("S3", min_value=0.0, max_value=500.0, value=100.0)
#     s4 = st.number_input("S4", min_value=0.0, max_value=500.0, value=100.0)
#     s5 = st.number_input("S5", min_value=0.0, max_value=500.0, value=100.0)
#     s6 = st.number_input("S6", min_value=0.0, max_value=500.0, value=100.0)

#     # Collect inputs into a numpy array
#     input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    

#     if st.button("Predict"):
#         # Make prediction
#         prediction = model.predict(input_data)

#         # Display the prediction
#         st.success(f"The predicted value is: {prediction[0]:.2f}")

# if __name__ == "__main__":
#     main()