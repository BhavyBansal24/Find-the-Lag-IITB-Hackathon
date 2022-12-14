# Importing Required Libraries
import streamlit as st # Library for WebApp
import numpy as np
import pandas as pd
import warnings
# Importing Sklearn required packages
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
# WebApp Setup
st.set_page_config(
    page_title="Find The Lag",
    page_icon = ":video_game:"
)
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

# Loading the dataset using pandas
dataset = pd.read_csv("Dataset.csv")
dataset = dataset.drop(['Serial Num'],axis=1)
X = dataset.iloc[:,:-1].values # Training Features
X_ = dataset.iloc[:,1:-1].values # Training Features
y = dataset.iloc[:,4].values # Target Variable

# Performing One-Hot Encoding
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# fit_transform "To use video_names as model_features"
X = np.array(columnTransformer.fit_transform(X))

# Train Test data split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state = 0)
X__train, X__test, y__train, y__test = train_test_split(X_,y,test_size = 0.1, random_state = 0)
#Linear Regression
linearRegressor = LinearRegression()
linearRegressor_ = LinearRegression()
linearRegressor.fit(X_train,y_train)
linearRegressor_.fit(X__train,y__train)
linearY_pred = linearRegressor.predict(X_test)
linearY_pred_ = linearRegressor_.predict(X__test)
linearAccuracy = linearRegressor.score(X_test,y_test)
linearAccuracy_ = linearRegressor_.score(X__test,y__test)

#Random Forest Regression
rfRegressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
rfRegressor_ = RandomForestRegressor(n_estimators = 300, random_state = 0)
rfRegressor.fit(X,y)
rfRegressor_.fit(X_,y)
rfY_pred = rfRegressor.predict(X_test)
rfY_pred_ = rfRegressor_.predict(X__test)
rfAccuracy = rfRegressor.score(X_test,y_test)
rfAccuracy_ = rfRegressor_.score(X__test,y__test)

# Code for WebApp Deployment 
st.title("Find the lag")
options = st.radio("Navigation", ("Dataset", "Lag Prediction Results", "Custom Input"), horizontal = True)
if options == "Dataset":
    option = st.selectbox("Options", ("Table", "Bar Chart"))
    if option == "Table":
        st.write(dataset)
    if option == "Bar Chart":
        print(dataset.columns)
        options = st.selectbox("Features", ("Video Name", "Frame Start", "Frame End", "Frame Difference"))
        st.bar_chart(dataset, x=options, y="Frame Time Difference (ms)")
if options == "Lag Prediction Results":
    ## Model without droping video names
    # col1, col2 = st.columns(2)
    # with col1:
    #     models = st.selectbox("Select Model", ("Linear Regression", "Random Forest Regression"))
    # with col2:
    #     st.write("\n")
    #     if models == "Linear Regression":
    #         st.write("Accuracy: ", linearAccuracy)
    #         st.write("MSE: ", float("{:.20f}".format(mean_squared_error(y_test, linearY_pred))))
    #     if models == "Random Forest Regression":
    #         st.write("Accuracy: ", rfAccuracy)
    #         st.write("MSE: ", float("{:.6f}".format(mean_squared_error(y_test, rfY_pred))))
    ## Model after droping video names
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Select Model", ("Linear Regression", "Random Forest Regression"))
    with col2:
        st.write("\n")
        if model == "Linear Regression":
            st.write("Accuracy: ", linearAccuracy_)
            st.write("MSE: ", float("{:.20f}".format(mean_squared_error(y__test, linearY_pred_))))
        if model == "Random Forest Regression":
            st.write("Accuracy: ", rfAccuracy_)
            st.write("MSE: ", float("{:.6f}".format(mean_squared_error(y__test, rfY_pred_))))
if options == "Custom Input":
    start_frame = st.text_input("Start Frame", 0)
    end_frame = st.text_input("End Frame", 0)
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    lag = end_frame - start_frame
    action = ["ADS HG", "Camera Right", "Crouch", "Melee", "Prone", "Reload HG", "Shoot AR", "Shoot HG", "Shoot LMG", "Walk Back"]
    video = st.selectbox("Select video name", (action))
    if video == "ADS HG":
        X = [1,0,0,0,0,0,0,0,0,0,start_frame,end_frame,lag]
    elif video == "Camera Right":
        X = [0,1,0,0,0,0,0,0,0,0,start_frame,end_frame,lag]
    elif video == "Crouch":
        X = [0,0,1,0,0,0,0,0,0,0,start_frame,end_frame,lag]
    elif video == "Melee":
        X = [0,0,0,1,0,0,0,0,0,0,start_frame,end_frame,lag]
    elif video == "Prone":
        X = [0,0,0,0,1,0,0,0,0,0,start_frame,end_frame,lag]
    elif video == "Reload HG":
        X = [0,0,0,0,0,1,0,0,0,0,start_frame,end_frame,lag]
    elif video == "Shoot AR":
        X = [0,0,0,0,0,0,1,0,0,0,start_frame,end_frame,lag]
    elif video == "Shoot HG":
        X = [0,0,0,0,0,0,0,1,0,0,start_frame,end_frame,lag]
    elif video == "Shoot LMG":
        X = [0,0,0,0,0,0,0,0,1,0,start_frame,end_frame,lag]
    elif video == "Walk Back":
        X = [0,0,0,0,0,0,0,0,0,1,start_frame,end_frame,lag]
    ftd_ = (end_frame - start_frame) * (1000/240)
    #ftd = "Frame Time Difference (using formula): " + str(ftd_)
    fd = "Frame Difference : " + str(end_frame - start_frame)
    st.info(fd)
    #st.info(ftd)
    models = st.selectbox("Select Model", ("Linear Regression", "Random Forest Regression"))
    pred_button = st.button("Predict")
    if pred_button:
        if (end_frame != 0):
            X = np.array(X)
            X = X.reshape(1,-1)
            linearY_pred = linearRegressor.predict(X)
            rfY_pred = rfRegressor.predict(X)
            if models == "Linear Regression" :
                LRP = "Predicted Frame Time Difference (ms) : " + str(linearY_pred[0])
                st.success(LRP)
            if models == "Random Forest Regression" :
                RFRP = "Predicted Frame Time Difference (ms) : " + str(rfY_pred[0])
                st.success(RFRP)
        else:
            st.error("Prediction unsuccessful. Please change values")
