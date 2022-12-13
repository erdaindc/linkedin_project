import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st
from PIL import Image

st.title("PREDICTING LINKEDIN USERS")
st.subheader("Eric Damp")
st.write("OPIM-607-201 Final Project")

#image = Image.open("C:\Users\edamp\my_project\Logo")

#with st.sidebar:
#    st.image(image)



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







#### 1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

#url="https://github.com/erdaindc/linkedin_project/blob/main/social_media_usage.csv"
#s = pd.read_csv(url)
s = pd.read_csv()


#### 3.	Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"]== 1,1,0),
    "income":np.where(s["income"] <10, s["income"],np.nan),
    "education":np.where(s["educ2"] <9, s["educ2"],np.nan),
    "parent":np.where(s["par"] ==1,1,0),
    "married":np.where(s["marital"] ==1,1,0),
    "female":np.where(s["gender"] ==2,1,0),
    "age":np.where(s["age"] <99, s["age"],np.nan)})

    
ss = ss.dropna()

#### 4.	Create a target vector (y) and feature set (X)

y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]


#### 5.	Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility



#### 6.	Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

lr = LogisticRegression(class_weight = "balanced")
lr.fit(x_train, y_train)


#### 7.	Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

y_pred = lr.predict(x_test)
confusion_matrix(y_test, y_pred)



#### 10.	Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

person = [income_sel,education_sel,parent_sel,
                    married_sel,female_sel,age_sel]

predict_linkedin = lr.predict([person])
probs = lr.predict_proba([person])

st.write(f"predicted class:{predict_linkedin[0]}")
st.write(f"Probability that this person has a LinkedIn account: {probs[0][1]}")


