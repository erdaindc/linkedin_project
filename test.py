import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st

st.title("PREDICTING LINKEDIN USERS")
st.subheader("Eric Damp")
st.write("OPIM-607-201 Final Project")




age_sel = st.number_input("Age", min_value=0, max_value=98, step=1, key=int)

female_sel = st.radio("Gender",
                    options=["Female","Male"])

if female_sel == "Female":
    female_sel = 1
else:
    female_sel = 0


married_sel = st.radio("Marital Status",
                    options=["Married", "Single"])

if married_sel == "Married":
    married_sel = 1
else:
    married_sel = 0


parent_sel = st.radio("Do you have a child under 18yrs",
                    options = ["Yes", "No"])


if parent_sel == "Yes":
    parent_sel = 1
else:
    parent_sel = 0


education_sel = st.selectbox("Your Education Level",
                options = ["Less than High School",
                "High School Incomplete",
                "High School Graduate",
                "Some College",
                "Associates Degree",
                "Bachelor's Degree",
                "Some Postgraduate Schooling",
                "Postgraduate Degree (Masters, Doctorate,etc)"])

if education_sel == "Less than High School":
    education_sel = 1
elif education_sel == "High School Incomplete":
    education_sel = 2
elif education_sel == "High School Graduate":
    education_sel = 3
elif education_sel == "Some College":
    education_sel = 4
elif education_sel == "Associates Degree":
    education_sel = 5
elif education_sel == "Bachelor's Degree":
    education_sel = 6
elif education_sel == "Some Postgraduate Schooling":
    education_sel = 7
elif education_sel == "Postgraduate Degree (Masters, Doctorate,etc)":
    education_sel = 8


income_sel = st.selectbox("Income",
                        options = ["Less than $10,000", "10 to under $20,000",
                        "20 to under $30,000","30 to under $40,000",
                        "40 to under $50,000","50 to under $75,000",
                        "75 to under $100,000","100 to under $150,000",
                        "$150,000+"])

if income_sel == "Less than $10,000":
    income_sel = 1
elif income_sel == "10 to under $20,000":
    income_sel = 2
elif income_sel == "20 to under $30,000":
    income_sel = 3
elif income_sel == "30 to under $40,000":
    income_sel = 4
elif income_sel == "40 to under $50,000":
    income_sel = 5
elif income_sel == "50 to under $75,000":
    income_sel = 6
elif income_sel == "75 to under $100,000":
    income_sel = 7
elif income_sel == "100 to under $150,000":
    income_sel = 8
elif income_sel == "150,000+":
    income_sel = 9






#### 10.	Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

new = ["income_sel","education_sel","parent_sel",
                    "married_sel","female_sel","age_sel"]

new["predict_linkedin"] = lr.predict(new)
new

probability = lr.predict_proba([new])

st.write(f"This person is {new} a LinkedIn user")
st.write(f"The probability that this person is a LinkedIn user is: {probs[0][1]}")