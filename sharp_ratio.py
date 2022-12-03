import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
#import plotly.graph_objects as go


def cumulative_return(stocks,choices):
    symbols, weights, investing_style, benchmark, rf, A_coef,ticker = choices.values()
    
    #tkers = sorted(set(stocks['Ticker'].unique()))
    #preprocess
    #data= stocks.copy()
    #print('data cumu',data)
    #data.set_index('Date',append=True)
    #df_by_stock= data.pivot(index='Date',columns='Ticker')
    #stocks = df_by_stock['Adj. Close']

    #stocks = data.pivot(index="Date", columns="Ticker", values="Adj. Close")
    data_stocks = stocks.copy()
    df = stocks.copy()
    df.set_index('Date', inplace=True)

    
    tkers = symbols.copy()
    logRet = np.log(df/df.shift())
    log_returns = np.log(df/df.shift())
    tickers_list = symbols.copy()
    weights_list = weights.copy()
    ##
    stock_port = {}
    for e in tickers_list: stock_port[e] = 0
    # Convert Weights to Floats and Sum
    weights = [float(x) for x in weights_list]
    s = sum(weights)
    # Calc Weight Proportions
    new_weights = []
    for i in weights: new_weights.append(i/s)
    # Assign Weights to Ticker Dict
    i = 0
    for e in stock_port:
        stock_port[e] = new_weights[i]
        i += 1
    
    port = dict.fromkeys(tkers, 0)
    port.update(stock_port)
    
    portfolio_dict = port
    
    for e in portfolio_dict:
        tmp = 0
        if portfolio_dict[e] > tmp:
            tmp = portfolio_dict[e]
            tick = e
    list_ =[]        
    for e in tickers_list:
        if e not in list_:
            list_.append(e)

    df = df[list_]
    df = df/df.iloc[0]
    df.reset_index(inplace=True)
    df=pd.DataFrame(df)
    fig = px.line(df, x='Date' ,y=df.columns[1:,])
    
    
    #layout reference = https://linuxtut.com/en/b13e3e721519c2842cc9/
    fig.update_layout(
        xaxis=dict(
        rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.update_layout(xaxis=dict(rangeselector = dict(font = dict( color = "black"))))
    fig.update_layout(title_text = 'Portfolio Historical Normalized Cumulative Returns',
                      title_x=0.458)
    st.plotly_chart(fig, use_container_width=True)
    
def sharp_ratio_func(df,choices):
    symbols, weights, investing_style, benchmark, rf, A_coef,ticker = choices.values()
    
    #tkers = sorted(set(stocks['Ticker'].unique()))
    #preprocess
    #data= stocks.copy()
    #df_by_stock= data.pivot(index='Date',columns='Ticker')
    #stocks = df_by_stock['Adj. Close']

    #stocks = data.pivot(index="Date", columns="Ticker", values="Adj. Close")
    stocks = df.copy()
    stocks.set_index('Date', inplace=True)
    tkers = stocks.columns
    tickers_list = symbols.copy()
    weights_list = weights.copy()
            
    stock_port = {}
    for e in tickers_list: stock_port[e] = 0
    # Convert Weights to Floats and Sum    
    weights = [float(x) for x in weights_list]
    s = sum(weights)
    # Calc Weight Proportions
    new_weights = []
    for i in weights: new_weights.append(i/s)
    # Assign Weights to Ticker Dict
    i = 0
    for e in stock_port:
        stock_port[e] = new_weights[i]
        i += 1
        
    port = dict.fromkeys(tkers, 0)
    port.update(stock_port)
        
    portfolio_dict = port
            
    sharp_ratio_list = []
    for ticker in symbols:
        logRet = np.log(stocks/stocks.shift())
        stk = dict.fromkeys(tkers, 0)
        stkTicker = {ticker:1}
        stk.update(stkTicker)
        ttlStk = np.sum(logRet*stk, axis=1)
        stock_sharpe_ratio = ttlStk.mean() / ttlStk.std()
        sharp_ratio_list.append(stock_sharpe_ratio)
            
    sharp_ratio = {'Assets': symbols, 'Sharpe Ratio': sharp_ratio_list}
        
        # Portfolio sharp Ratio Calculation
    logRet = np.log(stocks/stocks.shift())
    portfolio = dict.fromkeys(tkers, 0)
    portfolio.update(portfolio_dict)
    totalPortfolio = np.sum(logRet*portfolio, axis=1)
    portfolio_sharpe_ratio = totalPortfolio.mean() / totalPortfolio.std()
        
    sharp_ratio['Assets'].append('Portfolio')
    sharp_ratio['Sharpe Ratio'].append(portfolio_sharpe_ratio)
    
    fig = px.bar(sharp_ratio, x='Assets', y="Sharpe Ratio",color='Assets') 
    fig.update_layout(title_text = 'Sharpe Ratio of the Assets and Portfolio',
                        title_x=0.458)
    
    st.plotly_chart(fig, use_container_width=True)
        
