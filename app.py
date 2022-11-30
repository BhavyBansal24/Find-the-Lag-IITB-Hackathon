import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
warnings.simplefilter(action='ignore')

hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            </style>
            """
st.set_page_config(
    page_title="Find The Lag",
    page_icon = ":video_game:"
)
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)


dataset = pd.read_csv("Dataset.csv")
dataset = dataset.drop(['Serial Num'],axis=1)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X),dtype=np.str)
X = X[:, 1:] #Prevent Dummy Variable Trap
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state = 0)
    
#Linear Regression
linearRegressor = LinearRegression()
linearRegressor.fit(X_train,y_train)
linearY_pred = linearRegressor.predict(X_test)
linearAccuracy = linearRegressor.score(X_test,y_test)
    
#Random Forest Regression
rfRegressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
rfRegressor.fit(X,y)
rfY_pred = rfRegressor.predict(X_test)
rfAccuracy = rfRegressor.score(X_test,y_test)

st.title("Find the lag")
options = st.radio("Navigation", ("Dataset", "Lag Prediction Results"), horizontal = True)
if options == "Dataset":
    df = pd.read_csv("Dataset.csv")
    st.write(df)
if options == "Lag Prediction Results":
    models = st.selectbox("Select Model", ("Linear Regression", "Random Forest Regression"))
    if models == "Linear Regression":
        st.write("Linear Regression Accuracy: ", linearAccuracy)
        st.write("Random Forest Regression MSE: ", float("{:.20f}".format(mean_squared_error(y_test, linearY_pred))))
    if models == "Random Forest Regression":
        st.write("Random Forest Regression Accuracy: ", rfAccuracy)
        st.write("Random Forest Regression MSE: ", float("{:.6f}".format(mean_squared_error(y_test, rfY_pred))))