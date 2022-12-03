import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import plotly.express as px


def ER(stock_df, choices):
    symbols, weights, benchmark, investing_style, rf, A_coef,ticker  = choices.values()
    if benchmark == 'SP500':
        index_name ='sp500'
        index = pd.read_csv('sp500.csv')
    elif benchmark =='AOK':
        index_name ='AOK'
        index = pd.read_csv('AOK.csv')
    elif benchmark =='IXIC':
        index_name ='IXIC'
        index = pd.read_csv('IXIC.csv')

        
    tickers = symbols.copy()
    quantity = weights
    
    data_stocks = stock_df.copy()
    data_stocks.set_index('Date', inplace=True) 
    
    index = index.set_index(index.columns[0])

    #data_preprocess = data_preprocess[tickers]

    
    # merge with index
    df_copy =pd.merge(data_stocks, index, left_index=True, right_index=True)
    # geting index name
    index_name = []
    for name in index.columns: index_name= name
    # beta calculations
    log_returns = np.log(df_copy/df_copy.shift())
    cov = log_returns.cov()
    var = log_returns[index_name].var()
    
    beta_val = []
    for stock in tickers:
        beta=cov.loc[stock,index_name]/var
        beta_val.append(beta)
    df_beta = pd.DataFrame()
    df_beta['Tickers'] = tickers
    df_beta['Beta'] = beta_val  
    
    
    #calculating expected return 
    ER= []
    risk_free_return = 0.0138
    market_return = .105
  
    for beta in beta_val:
        expected_return = risk_free_return + beta*(market_return - risk_free_return)
        #print(expected_return)
        ER.append(expected_return)
    #print('ER',ER)
    #st.subheader('Expected Annual Return Based on CAPM Model')

    Expected_return = {'Assets': tickers, 'Expected Annual Return': ER}
    # Creates a header for streamlit
    #st.dataframe(Expected_return)

    
    # calculate expected return for the portfolio
    # portfolio weights assume equal
    portfolio_weights = []
    current_cash_value = 0
    total_portfolio_value = 0
    cash_value_stocks =[]
    for i in range(len(tickers) ):
        stocks_name = tickers[i]
        current_cash_value = df_copy[stocks_name].iloc[-1]
        stocks_quantity = quantity[i]
        cash_value = stocks_quantity * current_cash_value
        cash_value_stocks.append(cash_value)
        total_portfolio_value += cash_value
        portfolio_weights.append(cash_value)
    #print(portfolio_weights)
    portfolio_weights = (portfolio_weights / total_portfolio_value)*100
    ER_portfolio= []
    ER_portfolio = sum(list(ER) * portfolio_weights)/100
    #print(ER_portfolio)

    #st.subheader('Expected Portfolio Return Based on CAPM Model')
    # Creates a header for streamlit
    #st.write('Expected Portfolio Return is:', ER_portfolio)
    
    
    return beta_val, cash_value_stocks,Expected_return,ER_portfolio

def ER_graph(stock_df,choices):
    symbols, weights, benchmark, investing_style, rf, A_coef,ticker  = choices.values()
    beta,cash_value_weights,Expected_return,ER_portfolio = ER(stock_df,choices)
    
    Bar_output = Expected_return.copy()
    Bar_output['Assets'].append('Portfolio')
    Bar_output['Expected Annual Return'].append(ER_portfolio)
    fig = px.bar(Bar_output, x='Assets', y="Expected Annual Return",color='Assets') 
    fig.update_layout(title_text = 'Annual Expected Return of the Assets and Portfolio',
                      title_x=0.458)
    st.plotly_chart(fig, use_container_width=True)
    
    
def basic_portfolio(stock_df):
    """Uses the stock dataframe to graph the normalized historical cumulative returns of each asset.
    """
    # Calculates the daily returns of the inputted dataframe
    daily_return = stock_df.dropna().pct_change()
    # Calculates the cumulative return of the previously calculated daily return
    cumulative_return = (1 + daily_return).cumprod()

    
    # Graphs the cumulative returns
    st.line_chart(cumulative_return)


def display_heat_map(stock_df,choices):
    symbols, weights, benchmark, investing_style, rf, A_coef,ticker  = choices.values()
    selected_stocks = stock_df[symbols]
    # Calcuilates the correlation of the assets in the portfolio
    price_correlation = selected_stocks.corr()

    
    # Generates a figure for the heatmap
    fig, ax = plt.subplots()
    fig = px.imshow(price_correlation,text_auto=True, aspect="auto")
    # Displays the heatmap on streamlit
    st.write(fig)
   

#def display_portfolio_return(stock_df, choices):
    """Uses the stock dataframe and the chosen weights from choices to calculate and graph the historical cumulative portfolio return.
    """
#    symbols, weights, investment = choices.values()

    # Calculates the daily percentage returns of the 
#    daily_returns = stock_df.pct_change().dropna()
    # Applies the weights of each asset to the portfolio
#    portfolio_returns = daily_returns.dot(weights)
    # Calculates the cumulative weighted portfolio return
#    cumulative_returns = (1 + portfolio_returns).cumprod()
    # Calculates the cumulative profit using the cumulative portfolio return
#    cumulative_profit = investment * cumulative_returns

    # Graphs the result, and displays it with a header on streamlit
#    st.subheader('Portfolio Historical Cumulative Returns Based On Inputs!')
#    st.line_chart(cumulative_profit)
def buble_interactive(stock_df,choices):
    symbols, weights, benchmark, investing_style, rf, A_coef,ticker  = choices.values()
    beta,cash_value_weights,Expected_return,ER_portfolio = ER(stock_df,choices)
    my_list = []
    my_colors = []
    for i in beta:
        my_list.append(i)
        if i < 0.3:
            my_colors.append("Conservative")
        if i >= 0.3 and i <= 1.1:
            my_colors.append("Moderate Risk")
        if i > 1.1:
            my_colors.append("Risky")
    
    df_final =pd.DataFrame()
    df_final['ticker'] = symbols
    df_final['quantities'] = weights
    df_final['cash_value']  =cash_value_weights
    df_final['Beta'] = my_list
    df_final['Risk'] = my_colors
   
    fig = px.scatter(
    df_final,
    x="quantities",
    y="Beta",
    size="cash_value",
    color="Risk",
    hover_name="ticker",
    log_x=True,
    size_max=60,
    )
    fig.update_layout(title= benchmark +" Benchmark - Beta of Stock Ticker to Quantity")
    # -- Input the Plotly chart to the Streamlit interface
    st.plotly_chart(fig, use_container_width=True) 

    with st.container():
        st.header('Portfolio Health')

        average_comp = 0
        for i in df_final['Beta']:
            average_comp = average_comp + i
        average_comp = average_comp/df_final['Beta'].size
        average_comp = round(average_comp,2)


        st.write('You have selected to make your portfolio',investing_style,'. Refer to the following information below for more details on how the following portfolio compares to your investment style.')
        #Conservative investor message
        if investing_style == 'Conservative':
            if average_comp < 0.9:
                health = "Very Low Risk"
                st.write("Currently, your portfolio matches your investing style. The algorithm recommends making equal increases in your position.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
            if average_comp >= 0.9 and average_comp <= 1.1:
                health = "Balanced"
                suggestion = df_final.loc[df_final['Beta'] > 0.9, ['ticker']]
                x = suggestion.to_string(header=False,
                    index=False,
                    index_names=False).split('\n')
                vals = [','.join(ele.split()) for ele in x]
                print(vals)
                for i in range(0,len(vals)):
                    st.write("The algorithm recommends decreasing your postion in",vals[i],".")
                st.write("Having too many high risk stocks in your portfolio could significantly reduce your chances of making profitable returns.")
                st.write("It is important that your returns are balanced and do not contain too many stocks that are too volatile.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
            if average_comp > 1.1:
                health = "Risky"
                suggestion = df_final.loc[df_final['Beta'] >= 1.1, ['ticker']]
                x = suggestion.to_string(header=False,
                    index=False,
                    index_names=False).split('\n')
                vals = [','.join(ele.split()) for ele in x]
                print(vals)
                for i in range(0,len(vals)):
                    st.write("The algorithm recommends decreasing your postion in",vals[i],".")
                st.write("Having too many high risk stocks in your portfolio could significantly reduce your chances of making profitable returns.")
                st.write("It is important that your returns are balanced and do not contain too many stocks that are too volatile.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
        



        elif investing_style == 'Balanced':
            if average_comp < 0.9:
                health = "Very Low Risk"
                suggestion = df_final.loc[df_final['Beta'] < 0.9, ['ticker']]
                x = suggestion.to_string(header=False,
                    index=False,
                    index_names=False).split('\n')
                vals = [','.join(ele.split()) for ele in x]
                print(vals)
                for i in range(0,len(vals)):
                    st.write("The algorithm recommends decreasing your postion in",vals[i],".")
                st.write("Our algorithm recommend decreasing your postion in",suggestion,".")
                st.write("Having too many low risk stock in your portfolio could significantly reduce your chances of making profitable returns.")
                st.write("It is important that your return are able to keep up with the market conditions and the annual average inflation of 3%.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
            if average_comp >= 0.9 and average_comp <= 1.1:
                health = "Balanced"
                st.write("Currently, your portfolio matches your investing style. The algorithm recommends making equal increases in your position.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
            if average_comp > 1.1:
                health = "Risky"
                suggestion = df_final.loc[df_final['Beta'] >= 1.1, ['ticker']]
                x = suggestion.to_string(header=False,
                    index=False,
                    index_names=False).split('\n')
                vals = [','.join(ele.split()) for ele in x]
                print(vals)
                for i in range(0,len(vals)):
                    st.write("The algorithm recommends decreasing your postion in",vals[i],".")
                st.write("Having too many high risk stocks in your portfolio could significantly reduce your chances of making profitable returns.")
                st.write("It is important that your returns are balanced and do not contain too many stocks that are too volatile.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
                

       
       
       
       
        elif investing_style == "Risky":
            if average_comp < 0.9:
                health = "Very Low Risk"
                suggestion = df_final.loc[df_final['Beta'] < 0.9, ['ticker']]
                x = suggestion.to_string(header=False,
                    index=False,
                    index_names=False).split('\n')
                vals = [','.join(ele.split()) for ele in x]
                print(vals)
                for i in range(0,len(vals)):
                    st.write("The algorithm recommends decreasing your postion in",vals[i],".")
                st.write("Our algorithm recommend decreasing your postion in",suggestion,".")
                st.write("Having too many low risk stock in your portfolio could significantly reduce your chances of making profitable returns.")
                st.write("It is important that your return are able to keep up with the market conditions and the annual average inflation of 3%.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
            if average_comp >= 0.9 and average_comp <= 1.1:
                health = "Balanced"
                suggestion = df_final.loc[df_final['Beta'] < 1.1, ['ticker']]
                x = suggestion.to_string(header=False,
                    index=False,
                    index_names=False).split('\n')
                vals = [','.join(ele.split()) for ele in x]
                print(vals)
                for i in range(0,len(vals)):
                    st.write("The algorithm recommends decreasing your postion in",vals[i],".")
                st.write("Our algorithm recommend decreasing your postion in",suggestion,".")
                st.write("Having too many low risk stock in your portfolio could significantly reduce your chances of making profitable returns.")
                st.write("It is important that your return are able to keep up with the market conditions and the annual average inflation of 3%.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
            if average_comp > 1.1:
                health = "Risky"
                st.write("Currently, your portfolio matches your investing style. The algorithm recommends making equal increases in your position.")
                st.write("Your average beta is ", average_comp, "This puts your portfolio in a ", health, " Status.")
            