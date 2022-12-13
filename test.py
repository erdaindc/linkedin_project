st.header("Prediction of LinkedIn Users")
st.title("Eric Damp")
st.write("OPIM-607-201 Final Project")

age = st.number_input("Age", min_value=0, max_value=98, step=1, key=int)

female = st.radio("Gender",
                    options=["Female","Male"])

if female == "Female":
    female = 1
else:
    female = 0


married = st.radio("Marital Status",
                    options=["Married", "Single"])

if married == "Married":
    married = 1
else:
    married = 0


parent = st.checkbox("Do you have a child < 18yrs?",
                    options = ["Yes", "No"])


if parent == "Yes":
    parent = 1
else:
    parent = 0


education = st.selectbox("Your Education Level",
                options = ["Less than High School",
                "High School Incomplete",
                "High School Graduate",
                "Some College",
                "Associates Degree",
                "Bachelor's Degree",
                "Some Postgraduate Schooling",
                "Postgraduate Degree (Masters, Doctorate,etc)"])

if education == "Less than High School":
    education = 1
elif education == "High School Incomplete":
    education = 2
elif education == "High School Graduate":
    education = 3
elif education == "Some College":
    education = 4
elif education == "Associates Degree":
    education = 5
elif education == "Bachelor's Degree":
    education = 6
elif education == "Some Postgraduate Schooling":
    education = 7
elif education == "Postgraduate Degree (Masters, Doctorate,etc)":
    education = 8


income = st.selectbox("Income",
                        options = ["Less than $10,000", "10 to under $20,000",
                        "20 to under $30,000","30 to under $40,000",
                        "40 to under $50,000","50 to under $75,000",
                        "75 to under $100,000","100 to under $150,000",
                        "$150,000+"])

if income == "Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
elif income == "150,000+":
    income = 9