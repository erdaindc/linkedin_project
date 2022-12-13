import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st


st.selectbox("Education level",
                options = ["High SChool Diploma",
                            "GED",
                            "Graduate Degree"])