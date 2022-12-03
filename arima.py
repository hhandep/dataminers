import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import statsmodels.api as sm
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

# from sklearn.metrics import mean_squared_error

df = pd.read_csv('us-shareprices-daily.csv', sep=';')

def get_model_accuracy(data, ticker_symbol):
    
    stock_data = data[data['Ticker'] == ticker_symbol]


    # get MSE for testing data using 85/15 split for chosen stock symbol

    train_data, test_data = stock_data[0:int(len(stock_data)*0.85)], stock_data[int(len(stock_data)*0.85):]
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = sm.tsa.statespace.SARIMAX(history, order=(1,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)

    MSE_error = mean_squared_error(test_data, model_predictions)
    return 'Testing Mean Squared Error is {}'.format(MSE_error)


def arima_chart(df,choices):
    symbols, weights, benchmark, investing_style, rf, A_coef,ticker  = choices.values()
    #tickers = []
    #for i in ticker:
    #    if i != 'Portfolio':
    #        tickers.append(i)
    
    
    #ticker = tickers
    #fig = plt.figure() 
    fig = go.Figure()#make_subplots()
    #for ticker in tickers:
    ticker_df = pd.concat([df['Date'], df[ticker]], axis=1)
    model = sm.tsa.statespace.SARIMAX(ticker_df[ticker], order=(21,1,7))#21,1,7))
    model_fit = model.fit()#disp=-1)
    # print(model_fit.summary())
    forecast = model_fit.forecast(7, alpha=0.05)#.predict(start=1259, end=1289)
        #print(forecast)
    data = pd.DataFrame()
    data['Date'] = df['Date']
    data['Ticker'] = df[ticker]
    data['Forecast'] = forecast# model_fit.predict(start=1259, end=1289)
    #print(data)
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Ticker'],name='{} historical'.format(ticker)))
    fig.add_trace(go.Scatter(x=forecast.index,y=forecast,name='{} forecast'.format(ticker) ))
    
    #fig.update_layout(title="ARIMA forecast model for Selected Tickers")
    fig.update_layout(width=800,height=500)
    fig.update_layout(
        xaxis=dict(
        rangeselector=dict(
        buttons=list([
        dict(count=1,
        step="year",
        stepmode="backward"),
        ])),
        rangeslider=dict(
        visible=True
        ),))
    fig.update_layout(title_text = 'ARIMA forecast model for Selected Asset',
                        title_x=0.458)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text="Value")#, row=r, col=c)

    st.plotly_chart(fig,width=800,height=500)

    #https://blog.streamlit.io/make-your-st-pyplot-interactive/
