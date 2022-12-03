import pandas as pd
import numpy as np
from datetime import datetime as dt
from pypfopt.efficient_frontier import EfficientFrontier
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

### START AND RUN STREAMLIT
#https://docs.streamlit.io/library/get-started/installation

def ef_viz(stock_df,choices):
    #st.write("EF Visualization KOI EDITS")
    # st.header('CAPM Model and the Efficient Frontier')
    
    symbols, weights, benchmark, investing_style, rf, A_coef,ticker  = choices.values()
    tickers = symbols
    
    #tickers.append('sp500')
    #st.write(tickers)
    #st.write(stock_df)
    
    # Yearly returns for individual companies
    #https://stackoverflow.com/questions/69284773/unable-to-resample-the-pandas-with-date-column-typeerror-only-valid-with-dateti
    stock_dff = stock_df.copy()
    stock_dff['Date'] = pd.to_datetime(stock_dff['Date'])


    # ind_er_df = stock_dff.set_index('Date')
    #st.write(stock_dff.columns)
    ind_er_df = stock_dff.resample('Y', on = 'Date').last().pct_change().mean()
    ind_er = ind_er_df[tickers]
    #st.write(ind_er)
    ann_sd = stock_df[tickers].pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
    assets.columns = ['Returns', 'Volatility']
    assets
    #st.write(assets)
    ln_pct_change = stock_df[tickers].pct_change().apply(lambda x: np.log(1+x))[1:]
    #Cov Matrix
    cov_matrix =ln_pct_change.cov()

    ## CREATE PORFOLIOS WEIGHTS 
    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_assets = len(tickers)
    num_portfolios = 1000

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                        # weights 
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd)
        
    data = {'Returns':p_ret, 'Volatility':p_vol}

    for counter, symbol in enumerate(stock_df[tickers].columns.tolist()):
        #print(counter, symbol)
        data[symbol] = [w[counter] for w in p_weights]
        
    port_ef_df  = pd.DataFrame(data)
    port_ef_df['Vol'] = port_ef_df['Volatility']
    
    ## NEEDS INPUT INSTEAD OF HARD CODE
    #a = 5 #the coefficient of risk aversion is A. If an invest is less risk averse A is small. We assume 25 < A < 35. 
    #rf = 0.041
    
    min_vol_port = port_ef_df.iloc[port_ef_df['Volatility'].idxmin()]
    optimal_risky_port = port_ef_df.iloc[((port_ef_df['Returns']-rf)/port_ef_df['Volatility']).idxmax()]

    ### Make DF and data string for when hover over data points
    def make_op_df(df, tickers):
        new = {}
        op_str = str()
        new['Returns'] = df[0]
        new['Volatility'] = df[1]
        
        for i in range(0,len(tickers)):
            new[tickers[i]]= df[i+2]
            op_str += str(tickers[i]) + ': ' + str(round(df[i+2],4)) + '<br>'

        return pd.DataFrame(new, index=[0]), op_str

    op_df, op_str = make_op_df(optimal_risky_port, tickers)

    def make_port_str(df, tickers):
        port_str_lst = []
        for i in range(0,len(df)):
            temp = str()
            for u in range(0,len(tickers)):
                temp += str(tickers[u])+ ': ' + str(round(df[tickers[u]][i],4)) + '<br>'
            port_str_lst.append(temp)

        return port_str_lst

    port_str_lst = make_port_str(port_ef_df, tickers)   

    ## CREATE CAPM LINE #https://www.youtube.com/watch?v=JWx2wcrSGkk
    cal_x = []
    cal_y = []
    utl = []
    


    for er in np.linspace(rf, max(data['Returns'])+rf,20):
        sd = (er - rf)/ ((optimal_risky_port[0] - rf)/ optimal_risky_port[1])
        u = er - 0.5*A_coef*(sd**2)
        cal_x.append(sd)
        cal_y.append(er)
        utl.append(u)
    
    data2 = {'Utility':utl, 'cal_x':cal_x, 'cal_y':cal_y}

    utl_df  = pd.DataFrame(data2)
    
    ## Create Figure
    fig3 = go.Figure()

    #https://plotly.com/python/colorscales/
    fig3.add_trace(go.Scatter(x=port_ef_df['Volatility'], y=port_ef_df['Returns'], hovertemplate='Volatility: %{x} <br>Returns: %{y} <br>%{text}',\
                                text= port_str_lst, mode='markers', \
                                marker=dict(color=port_ef_df['Volatility'],  colorbar=dict(title="Volatility"), \
                                size=port_ef_df['Returns']*50, cmax=max(port_ef_df['Volatility']),\
                                cmin=min(port_ef_df['Volatility'])),name='Portfolio'))
                                #, mode='markers', size=port_ef_df['Returns'], \
                                    #size_max=30, color=port_ef_df['Vol']))
    fig3.add_trace(go.Scatter(x=utl_df['cal_x'], y=utl_df['cal_y'], mode='lines', line = dict(color='rgba(11,156,49,1)'),name='Ultility Function',\
                                hovertemplate='Volatility: %{x} <br>Returns: %{y}')) #))

    fig3.add_trace(go.Scatter(x=op_df['Volatility'], y=op_df['Returns'], mode='markers', \
                        marker=dict(color= 'rgba(11,156,49,1)', size=30),\
                        hovertemplate='Volatility: %{x} <br>Returns: %{y} <br>%{text}',\
                        text=[op_str]))
    ### HOVER TEMPLATE # https://plotly.com/python/hover-text-and-formatting/
            
   
#     ### SAVE IN CASE CANNOT FIGURE OUT THE HOVER TEMPLATE
#     fig2 = px.scatter(op_df, 'Volatility', 'Returns')
#     fig2.update_traces(marker=dict(color= 'rgba(11,156,49,1)', size=35))

#     fig1 = px.line(utl_df, x="cal_x", y="cal_y")
#     #fig1.update_traces(line=dict(color = 'rgba(11,156,49,1)'))
    
#     fig = px.scatter(port_ef_df, 'Volatility', 'Returns', size='Returns', size_max=30, color='Vol')
# #https://stackoverflow.com/questions/59057881/python-plotly-how-to-customize-hover-template-on-with-what-information-to-show
# #https://stackoverflow.com/questions/65124833/plotly-how-to-combine-scatter-and-line-plots-using-plotly-express

#     #data3 = 
#     fig3.data = [fig2.data,fig1.data,fig.data]
#     #fig3.update_traces(line=dict(color = 'rgba(11,156,49,1)'))
#     ####

    fig3.update_layout(showlegend=False)#, legend_title_text = "Contestant")
    fig3.update_xaxes(title_text="Volatility")
    fig3.update_yaxes(title_text="Portfolio Return Rates")

    st.plotly_chart(fig3, use_container_width=True) 

    #st.write(op_str)
    op_df = op_df.style.set_properties(**{'color':'green'})
    st.subheader('Optimal Returns vs Volatility and Portfolio weights')
    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")

    with col2:
        st.write(op_df)

    with col3:
        st.write("")
    
    im = Image.open('EFvsMinvar.png')
    st.subheader('Understand the Efficient Frontier')
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image(im, caption='Elements of the Efficient Frontier',use_column_width='auto')

    with col3:
        st.write("")
    