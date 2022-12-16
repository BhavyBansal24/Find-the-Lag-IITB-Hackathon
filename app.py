import streamlit as st
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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
X = np.array(columnTransformer.fit_transform(X))
# X = X[:, 1:] #Prevent Dummy Variable Trap
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
options = st.radio("Navigation", ("Dataset", "Lag Prediction Results", "User Values"), horizontal = True)
if options == "Dataset":
    df = pd.read_csv("Dataset.csv")
    st.write(dataset)
if options == "Lag Prediction Results":
    models = st.selectbox("Select Model", ("Linear Regression", "Random Forest Regression"))
    if models == "Linear Regression":
        st.write("Accuracy: ", linearAccuracy)
        st.write("MSE: ", float("{:.20f}".format(mean_squared_error(y_test, linearY_pred))))
    if models == "Random Forest Regression":
        st.write("Accuracy: ", rfAccuracy)
        st.write("MSE: ", float("{:.6f}".format(mean_squared_error(y_test, rfY_pred))))
if options == "User Values":
    start_frame = st.text_input("Start Frame", 0)
    end_frame = st.text_input("End Frame", 0)
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    lag = end_frame - start_frame
    ftd_ = (end_frame - start_frame) * (1000/240)
    ftd = "Frame Time Difference : " + str(ftd_)
    fd = "Frame Difference : " + str(end_frame - start_frame)
    action = ["ADS HG", "Camera Right", "Crouch", "Melee", "Prone", "Reload HG", "Shoot AR", "Shoot HG", "Shoot LMG", "Walk Back"]
    video = st.selectbox("Select video", (action))
    if video == "ADS HG":
        X = [1,0,0,0,0,0,0,0,0,0,start_frame,end_frame,lag]
    if video == "Camera Right":
        X = [0,1,0,0,0,0,0,0,0,0,start_frame,end_frame,lag]
    if video == "Crouch":
        X = [0,0,1,0,0,0,0,0,0,0,start_frame,end_frame,lag]
    if video == "Melee":
        X = [0,0,0,1,0,0,0,0,0,0,start_frame,end_frame,lag]
    if video == "Prone":
        X = [0,0,0,0,1,0,0,0,0,0,start_frame,end_frame,lag]
    if video == "Reload HG":
        X = [0,0,0,0,0,1,0,0,0,0,start_frame,end_frame,lag]
    if video == "Shoot AR":
        X = [0,0,0,0,0,0,1,0,0,0,start_frame,end_frame,lag]
    if video == "Shoot HG":
        X = [0,0,0,0,0,0,0,1,0,0,start_frame,end_frame,lag]
    if video == "Shoot LMG":
        X = [0,0,0,0,0,0,0,0,1,0,start_frame,end_frame,lag]
    if video == "Walk Back":
        X = [0,0,0,0,0,0,0,0,0,1,start_frame,end_frame,lag]
    st.success(fd)
    st.success(ftd)
    X = np.array(X)
    X = X.reshape(1,-1)
    linearY_pred = linearRegressor.predict(X)
    rfY_pred = rfRegressor.predict(X)
    LRP = "Linear Regression Prediction : " + str(linearY_pred[0])
    st.success(LRP)
    RFRP = "Random Forest Regression Prediction : " + str(rfY_pred[0])
    st.success(RFRP)
