import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as pdr
from datetime import datetime



def convert_simFin2(path):
    df = pd.read_csv(path, sep=';')
    stocks = df.pivot(index="Date", columns="Ticker", values="Adj. Close")
    return stocks

def log_of_returns2(stocks):
    log_returns = np.log(stocks/stocks.shift())
    return log_returns





# Code to Calculate and output Betas
# Read in Stock csv data and convert to have each Ticker as a column.
#df = pd.read_csv('D:/SimFinData/us-shareprices-daily.csv', sep=';')
#stocks = df.pivot(index="Date", columns="Ticker", values="Adj. Close")
#stocks
#start = min(df['Date'])
#end = max(df['Date'])
#logRet = np.log(stocks/stocks.shift())


#SP500 = pdr.get_data_yahoo("^GSPC", start)
#IXIC = pdr.get_data_yahoo("^IXIC", start)
#AOK = pdr.get_data_yahoo("AOK", start)

#SP500['SP500'] = SP500['Adj Close']
#IXIC['IXIC'] = IXIC['Adj Close']
#AOK['AOK'] = AOK['Adj Close']

#spAC = np.log(SP500['SP500']/SP500['SP500'].shift())
#spAC = spAC.loc[spAC.index <= end]

#ixicAC = np.log(IXIC['IXIC']/IXIC['IXIC'].shift())
#ixicAC = ixicAC.loc[ixicAC.index <= end]

#aokAC = np.log(AOK['AOK']/AOK['AOK'].shift())
#aokAC = aokAC.loc[aokAC.index <= end]

#sp500B = logRet.join(spAC)
#ixicB = logRet.join(ixicAC)
#aokB = logRet.join(aokAC)

#sp5Cov = sp500B.cov()
#ixicCov = ixicB.cov()
#aokCov = aokB.cov()

#sp500Var = sp500B['SP500'].var()
#ixicVar = ixicB['IXIC'].var()
#aokVar = aokB['AOK'].var()

#sp500Beta = sp5Cov.loc['SP500']/sp500Var
#ixicBeta = ixicCov.loc['IXIC']/ixicVar
#aokBeta = aokCov.loc['AOK']/aokVar

#betas = pd.concat([sp500Beta,ixicBeta,aokBeta], axis=1)

#betas['Ticker'] = betas.index

#betas = betas[['Ticker','SP500','IXIC','AOK']]

#betas.to_csv (r'betas.csv', index = None, header=True)



