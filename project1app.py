# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import numpy as np
import streamlit as st 
from pickle import load
from sklearn.preprocessing import RobustScaler
import plotly.express as px








data=pd.read_csv("clean_data.csv",index_col=0)
#array = data.values

X = data.iloc[:,1:] #array[:, 2:]
scaler = RobustScaler()
scaler.fit(X)
df = pd.DataFrame(X)
loaded_model = load(open('xgbclf.sav', 'rb'))





nav = st.sidebar.radio("Navigation",["EDA","Predict Churn"])


if nav == "Predict Churn":
         
    # creating a function for Prediction
    
    def churn_prediction(input_data):
     
     
         # changing the input_data to numpy array
         #input_data_as_numpy_array = np.asarray(input_data)
         new_input = pd.DataFrame([input_data])
         
         # reshape the array as we are predicting for one instance
         #input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
         
         input_data_reshaped=scaler.transform(new_input)#(input_data_reshaped)
         new_data = pd.DataFrame(input_data_reshaped)
         prediction = loaded_model.predict(new_data)
         print(prediction)
         
         y_pred = loaded_model.predict_proba(input_data_reshaped)[:,1]
         churn_probs = y_pred[:1]*100
         
         
         if (prediction == 1):
           #return 'Churn '
           st.write("Probability of Churn is", round(churn_probs[0],2),"%" )
           st.error("Hence Customer will churn :thumbsdown: ")
         else:
           #return 'Did not Churn '
           st.write("Probability of Churn is", round(churn_probs[0],2),"%")
           st.success("Hence customer will not churn :thumbsup: ")
                      
     
    
                    
    def main():
     
        # giving a title
        st.title('Model Deployment: XGBoost Model')
        
        
        # getting the input data from the user
        
        
        number1 = st.number_input('Insert  Duration of Account')
        number2 = st.number_input('Insert  Voice Plan')
        number3 = st.number_input('Insert  no. of Voice Messages')
        number4 = st.number_input('Insert  International Plan')
        number5 = st.number_input('Insert  International Minutes')
        number6 = st.number_input('Insert  Total International Calls')
        number7 = st.number_input('Insert  Total International Charge')
        
        number8 = st.number_input('Insert  Total number of calls during the day')
        number9 = st.number_input('Insert  Total number of calls during the evening')
        number10 = st.number_input('Insert Total number of calls during the night')
        number11 = st.number_input('Insert Number of calls to customer service')
        number12 = st.number_input('Insert Total_Charge')
        
        
        
        #     # code for Prediction
        diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Churn Result'):
            diagnosis = churn_prediction([number1, number2, number3,              number4,number5,number6,number7,number8,number9,number10,number11,number12])
        
        
        #st.success(diagnosis)
        
    if __name__ == '__main__':
        main()
        
if nav == "EDA":
    st.subheader("Data")
    st.write(data)   
    
    old_df = pd.read_csv("Churn.csv")
    
    bar_chart1 = px.bar(old_df, x="state", y="churn", color=("state"))
    st.write(bar_chart1)
    
    
    bar_chart2 = px.bar(old_df, x="state", y="intl.mins", color=("state"))#, animation_frame=("intl.mins"), animation_group=("state"))
    st.write(bar_chart2)
    
    #bar_chart3 = px.bar(old_df, x="intl.calls", y="churn", color=("intl.calls"))
    #st.write(bar_chart3)
    
    
    #st.line_chart(data)
    
    #st.area_chart(data)
    
    
    
    

    


