import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as pdr

# Read in Stock csv data and convert to have each Ticker as a column.
#df = pd.read_csv('us-shareprices-daily.csv', sep=';')
#stocks = df.pivot(index="Date", columns="Ticker", values="Adj. Close")
#logRet = np.log(stocks/stocks.shift())

# Calculate the Correlation Coefficient for all Stocks
#stocksCorr = logRet.corr()

# Output to csv
#stocksCorr.to_csv (r'correlation_matrix.csv', index = None, header=True)

# Enter path of SimFin Data to convert to format for Calculations
def convert_simFin(path):
    df = pd.read_csv(path, sep=';')
    stocks = df.pivot(index="Date", columns="Ticker", values="Adj. Close")
    return stocks

# Calculate Log returns of the Formatted Stocks
def log_of_returns(stocks):
    log_returns = np.log(stocks/stocks.shift())
    return log_returns

# Enter Log returns of Stocks to Calculate the Correlation Matrix.
def correlation_matrix(lr):
    return lr.corr()
    