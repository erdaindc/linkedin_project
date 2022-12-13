# Final Project
## Eric Damp
### December 10, 2022

import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#### 1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

s = pd.read_csv("social_media_usage.csv")
s.shape

***

#### 2.	Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

def clean_sm (x):
    x = np.where(x == 1,
            1,
            0)  
    return(x)

tdf = {
    'col_a': [1,5,3],
    'col_b': [1,1,0]
}

tdf = pd.DataFrame(tdf)

clean_sm(tdf)

***

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

ss.describe
ss.isna().any()
ss.info()

ss.boxplot(column=["income","education","parent","married","female"])

***

#### 4.	Create a target vector (y) and feature set (X)

y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]

***

#### 5.	Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# The x_train and y_train contain 80% of the data we will use to train the model on. The x_train contains the features used to
# predict the y as we train the model.

# The x_test and y_test include the remaining 20% of the dataset. Similar to the above, x_test includes the features that will
# be used to predict the y_test when we TEST the model we generated with data we have not seen or touched yet. This will 
# help evaluate the model's performance.


***

#### 6.	Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

lr = LogisticRegression()
lr.fit(x_train, y_train)

***

#### 7.	Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

y_pred = lr.predict(x_test)
confusion_matrix(y_test, y_pred)

accuracy = ((145+34)/(145+23+50+34))
accuracy

#In the confusion matrix, the upper left number is the value that the model predicted would be negative, and actually is negative.
#The lower right number is what the model predicted is true (linkedin user) that actually was true. Taken together, the 
#upper left and lower right represent how accurate the model classified predicted values.
#The lower left is the number of false negative - the model predicted it was negative but actually was positive.
#The upper right shows how many false positives - values where the model predicted it was positive, but was actually negative (not a Linkedin user)

***

#### 8.	Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
                           columns=["Predicted negative", "Predicted positive"],
                           index=["Actual negative","Actual positive"]).style.background_gradient(cmap="Spectral")

conf_matrix

***

#### 9.	Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

precision = 34/(34+23) #0.5964 = 0.596
precision  
#Precision measures how many of the samples predicted as positive actually are positive. This measurement is useful when
#you are trying to minimize the number of false positives. An example here would be during COVID, when you had to get
#a COVID test to fly, go to work, etc. You wanted to use a test that was very precise.

recall = 34/(34+50) #0.4047 = 0.405
recall  
#Recall is when you are trying to capture all positives (true, predicted, and false negatives). It is the same as the
#sensitivity that is calculated in R for logistic regression. Essentially, how good is it at categorizing the positives?
#This would be very helpful in diagnosing rare disorders or potentially life-threatening illnesses. You would want to undergoe
#a test that is sensitive enough to detect the disease.

f1 = ((0.596 * 0.405)/(0.596 + 0.405))*2     #0.482
f1  
#The F1 factors in both precision and recall. Whereas the two above measurements tell one side of the story, the F1
#takes both into account. If you have imbalanced binary classificaiton datasets, the F1 can be a better barometer than
#looking at the accuracy score.

print(classification_report(y_test, y_pred))

***

#### 10.	Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

new = pd.DataFrame({
    "income":[8,8],
    "education":[7,7],
    "parent":[0,0],
    "married":[1,1],
    "female":[1,1],
    "age":[42,82]})

new["predict_linkedin"] = lr.predict(new)
new

#The model predicts the 42yr old person has a LinkedIn account, whereas the 82yr old does not

***

#### URL FOR STREAMLIT
# FILL THIS IN WHEN READY

