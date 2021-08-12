# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:49:24 2021

@author: hemanth
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import datetime 
#from datetime import datetime, timedelta
#import time

model = keras.models.load_model("model.h5")


def welcome():
    return "Welcome All"


def predict_price(final_features):
    pred_price = model.predict(final_features)
    return pred_price


def main():
    st.title("Tesla Inc. Stock Price Prediction")
    html_temp = """
    <div style="background-color:rgb(0, 238, 255);padding:10px">
    <h2 style="color:rgb(255, 124, 37);text-shadow: 0 4px 10px rgba(0, 0, 0, 0.603);text-align:center;">Tesla Inc. Predicted CLosed Price</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    #dateparse = lambda dates: [pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
    #dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    #stock_data = pd.read_csv(r'C:\Users\hemua\Desktop\stock\stock_data.csv', parse_dates=['Date'], date_parser=dateparse)
    stock_data = pd.read_csv('stock_dat.csv' , parse_dates=['Date'])
    #stock_data['Date'] = pd.Timestamp(stock_data['Date'])
    #stock_data['Date'] = pd.to_datetime(stock_data['Date'], format="%m/%d/%y",infer_datetime_format=True)
    #print(stock_data.dtypes)
    stock_data=stock_data.set_index("Date")
    X = stock_data['Close']
    

    # Getting the start day and next day from the dataset
    #stock_data.sort_values(by="Date",ignore_index=True,inplace=True)
    
    start_day = stock_data.index[0]
    last_day = stock_data.index[-1]
    next_day = last_day + datetime.timedelta(days = 1)
    


    # Taking date input
    input_date = st.date_input("Enter a Date: ", next_day)
    # Updating Date input
    input_date = datetime.datetime.strptime(str(input_date) , '%Y-%m-%d')
    #input_date = datetime.datetime.strptime(str(input_date) + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    #print(type(input_date))

    if input_date <= next_day and input_date >= start_day + datetime.timedelta(days=40):

        scaler = MinMaxScaler(feature_range=(0, 1))

        # Create a list of dates from the stock_data and get the index of the input date
        dates_list = []
        for dt in stock_data.index:
            dates_list.append(str(dt))
        

        j = 1
        
        while str(input_date - datetime.timedelta(days=j)) not in dates_list:
            j += 1
    
        i = dates_list.index(str(input_date - datetime.timedelta(days=j)))

        X = stock_data.filter(['Close'])
        #print(i)
        # Get the last 40 day closing price values and convert the dataframe to an array
        last_40_days = X[i - 40: i].values
        
        # Scale the data to be values between 0 and 1
        last_40_days_scaled = scaler.fit_transform(last_40_days)
        # Create an empty list
        X_test = []
        # Append the past 40 days
        X_test.append(last_40_days_scaled)
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Predict the Close Price
        result = 0
        if st.button("Predict"):
            result = predict_price(X_test)

        # undo the scaling
        result = np.array(result).reshape(1, -1)
        pred_price = scaler.inverse_transform(result)
        #print(pred_price)

        st.success("Predicted Close Price for {} is ${}".format(input_date, str(pred_price)[2:-2]))

        # Percentage increase or decrease in Closed Price
        #previous = pred_price
        
        
        #previous_pred_price = X.at[str(input_date - datetime.timedelta(days=j)), 'Close']
        previous_pred_price = X[i : i+1].values
        #print(previous_pred_price)
        #print(previous_pred_price1)
        diff = (float)(pred_price - previous_pred_price)
        #print( diff )
        if (diff < 0):
            a=(str(np.round(((- (diff) / previous_pred_price) * 100), 2)))
            a=str(a)[2:-2]
            st.write("percentage decrease = ", a)
        else:
            a=(str(np.round((( (diff) / previous_pred_price) * 100), 2)))
            a=str(a)[2:-2]
            st.write("percentage increase = ", a))
        #a=(str(np.round((( (diff) / previous_pred_price) * 100), 2)))
        #st.write("percentage = ", a)
    else:
        st.error(
            'Error: Either the date is above the last date of the dataset OR below the start date + 40 days of the dataset. Please enter a date between or equal to {} and {} !!'.format(
                start_day + datetime.timedelta(days=40), next_day))


if __name__ == '__main__':
    main()
