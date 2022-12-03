import streamlit as st
from datetime import date, timedelta
#from rest_api.fetch_data import (get_symbol_data)
import pandas as pd
from PIL import Image
import time

from plots import (
    display_heat_map,
    ER,
    buble_interactive,
    ER_graph
)

### Koi
from ef import(
    ef_viz
)
def risk_str(num):
    if num >=5 and num <15:
        return 'Low Risk Aversion'
    elif num >= 15 and num <25:
        return 'Medium Risk Aversion'
    elif num >= 25 and num <=35:
        return 'High Risk Aversion'
#### Koi
        
from sharp_ratio import(
    cumulative_return,
    
    sharp_ratio_func
)

from arima import (
    # get_model_accuracy,
    arima_chart
)


        
def load_heading():
    """The function that displays the heading.
        Provides instructions to the user
    """
    with st.container():
        st.title('Dataminers')
        header = st.subheader('This App performs historical portfolio analysis and future analysis ')
        st.subheader('Please read the instructions carefully and enjoy!')
        # st.text('This is some text.')


def get_choices():
    """Prompts the dialog to get the All Choices.
    Returns:
        An object of choices and an object of combined dataframes.
    """
    choices = {}
    #tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tickers", "Quantity", "Benchmark","Risk Free Return","Risk Aversion"])

    tickers = st.sidebar.text_input('Enter stock tickers.', 'GOOG,AA,AVGO,AMD')

    # Set the weights
    weights_str = st.sidebar.text_input('Enter the investment quantities', '50,30,25,25')

    benchmark = st.sidebar.selectbox(
        'Select your ideal benchmark of return',
        ('SP500', 'AOK', 'IXIC'))
    if benchmark == 'IXIC':
        st.sidebar.warning("You have selected a volatile benchmark.")
    elif benchmark == 'SP500':
        st.sidebar.success('You have selected a balanced benchmark')
    elif benchmark == 'AOK':
        st.sidebar.success('You have selected a conservative benchmark')

    ### koi
    rf = st.sidebar.number_input('Enter current rate of risk free return', min_value=0.001, max_value=1.00, value=0.041)


    #A_coef_map = 
    A_coef = st.sidebar.slider('Enter The Coefficient of Risk Aversion', min_value=5, max_value=35, value=30, step=5)

    if A_coef > 20:
        st.sidebar.success("You have selected a "+ risk_str(A_coef) +" investing style")
        investing_style = 'Conservative'
    elif A_coef >10 and A_coef <= 20:
        st.sidebar.success("You have selected a "+risk_str(A_coef) +" investing style")
        investing_style = 'Balanced'
    elif A_coef <= 10:
        st.sidebar.warning("You have selected a "+ risk_str(A_coef) +" investing style")
        investing_style = 'Risky'

    # arima model ticker    
    selected_tickers= []
    ticker_select = tickers.split(",")
    selected_tickers.extend(ticker_select)
    ticker = st.selectbox("Select a ticker to apply ARIMA Model",selected_tickers)

    # Every form must have a submit button.
    submitted = st.sidebar.button("Calculate")

    symbols = []
    reset = False

    # Reusable Error Button DRY!
    #def reset_app(error):
    #    st.sidebar.write(f"{error}!")
    #    st.sidebar.write(f"Check The Syntax")
    #    reset = st.sidebar.button("RESET APP")

    if submitted:
        #with st.spinner('Running the calculations...'):
        #    time.sleep(8)
        #    st.success('Done!')
        # convert  strings to lists
        tickers_list = tickers.split(",")
        weights_list = weights_str.split(",")
        #crypto_symbols_list = crypto_symbols.split(",")
        # Create the Symbols List
        symbols.extend(tickers_list)
        #symbols.extend(crypto_symbols_list)
        # Convert Weights To Decimals
    
        weights = []
        for item in weights_list:
            weights.append(float(item))

        if reset:
        #    # Clears all singleton caches:
            #tickers = st.sidebar.selectbox('Enter 11 stock symbols.', ('GOOG','D','AAP','BLK'))
        #    crypto_symbols = st.sidebar.text_input('Enter 2 crypto symbols only as below', 'BTC-USD,ETH-USD')
            #weights_str = st.sidebar.text_input('Enter The Investment Weights', '0.3,0.3 ,0.3')

            st.experimental_singleton.clear()


        else:
            # Submit an object with choices
            choices = {

                'symbols': symbols,
                'weights': weights,
                'benchmark': benchmark,
                'investing_style': investing_style,
                'risk-free-rate': rf,
                'A-coef': A_coef,
                'ticker':ticker

            }
            # Load combined_df
            data = pd.read_csv('data_and_sp500.csv')
            combined_df = data[tickers_list]
            raw_data=pd.read_csv('us-shareprices-daily.csv', sep=';')
            sharpe_data =pd.read_csv('sharpe_data.csv')
            # return object of objects
            return {
                'choices': choices,
                'combined_df': combined_df,
                'data': data,
                'raw_data':raw_data,
                'sharpe_data':sharpe_data
            }


def run():
    """The main function for running the script."""

    load_heading()
    choices = get_choices()
    if choices:
        st.success('''** Selected Tickers **''')  
        st.header('Capital Asset Pricing Model(CAPM)')
        """
The Capital Asset Pricing Model(CAPM) calculates the relationship between risk and expected return for the assets 
for the selected stocks for your portfolio. This model establishes a linear relationship between the return on an investment
and risk. CAPM model is based on the relationship between an asset's beta, risk-free rate,
and the expected return on the market minus the risk-free rate. 
                
Beta Values:\n
The beta is a way to measure the investment risk. It measure of how much risk the investment 
will add to a portfolio. If a stock is riskier than the market, it will have a beta greater than one.
if a stock has a beta of less than one, it will reduce the risk of the portfolio. 
        """
        
        with st.spinner('Running CAPM Beta Calculations...'):
            buble_interactive(choices['sharpe_data'],choices['choices'])
        #st.header('Tickers Beta')
        """
Expected Return:\n       
        """
        with st.spinner('Running CAPM Expected Return Calculations...'):

            ER_graph(choices['sharpe_data'], choices['choices'])

        
        ##### EDIT HERE ##### koi
        st.header('CAPM Model and the Efficient Frontier')
        """
    CAPM model measures systematic risks, however many of it's functions have unrealistic assumptions and rely heavily on a linear interpretation 
    of the risks vs. returns relationship. It is better to use CAPM model in conjunction with the Efficient Frontier to better
    graphically depict volatility (a measure of investment risk) for the defined rate of return. \n
    Below we map the linear Utility function from the CAPM economic model along with the Efficient Frontier
    Each circle depicted above is a variation of the portfolio with the same input asset, only different weights. 
    Portfolios with higher volatilities have a yellower shade of hue, while portfolios with a higher return have a larger radius. \n
    As you input different porfolio assets, take note of how diversification can improve a portfolio's risk versus reward profile.
        """
        with st.spinner('Running Efficient Frontier Model...'):

            ef_viz(choices['data'],choices['choices'])
        """
    There are in fact two components of the Efficient Frontier: the Efficient Frontier curve itself and the Minimum Variance Frontier. 
    The lower curve, which is also the Minimum Variance Frontier will contain assets in the portfolio 
    that has the lowest volatility. If our portfolio contains "safer" assets such as Governmental Bonds, the further to the right 
    of the lower curve we will see a portfolio that contains only these "safe" assets, the portfolios on 
    this curve, in theory, will have diminishing returns.\n
    The upper curve, which is also the Efficient Frontier, contains portfolios that have marginally increasing returns as the risks 
    increases. In theory, we want to pick a portfolio on this curve, as these portfolios contain more balanced weights of assets 
    with acceptable trade-offs between risks and returns. \n
    If an investor is more comfortable with investment risks, they can pick a portfolio on the right side of the Efficient Frontier. 
    Whereas, a conservative investor might want to pick a portfolio from the left side of the Efficient Frontier. \n
    Take notes of the assets' Betas and how that changes the shape of the curve as well. \n 
    How does the shape of the curve change when 
    the assets are of similar Beta vs when they are all different?\n 
    Note the behavior of the curve when the portfolio contains only assets with Betas higher than 1 vs. when Betas are lower than 1.\n  

        """
        ##### ##### Koi
        # Creates the title for streamlit
        st.subheader('Portfolio Historical Normalized Cumulative Returns')
        """
Cumulative Returns:\n 
The cumulative return of an asset is calculated by subtracting the original price paid from the current profit or loss. This answers the question, 
what is the return on my initial investment?\n 
The graph below shows the historical normalized cumulative returns for each of the chosen assets for the entire time period of the available data. 
The default line chart shows tickers AA, AMD, AVGO, and GOOG and we can see that all have a positive cumulative return over the period of the available data. 
Any of these assets purchased on the starting day and sold on the ending day for the period would have earned a return on their investment.\n
This chart can also be used to analyze the correlation of the returns of the chosen assets over the displayed period. 
Any segments of the line charts that show cumulative returns with similarly or oppositely angled segments can be considered to have some level of 
correlation during those periods. 
        """
        cumulative_return(choices['sharpe_data'], choices['choices'])
        """
Negative Correlations (1): \n
Occur for assets whose cumulative returns move in opposite directions. When one goes up the other goes down and vice versa. 
These negatively correlated assets would offer some level of diversification protection to each other. 
Perfectly negatively correlated stocks are sort of the goal, but unlikely to be common. 
In most cases finding some level of negatively correlated stocks, should offer some level of diversification protection to your portfolio. 
The amount of protection depends upon the calculated metric. Our tool includes some CAPM analysis, which attempts to relate the risk and return 
and the correlation of assets to determine the expected portfolio returns versus the combined, hopefully reduced, risk.\n

Positive Correlations (2):\n 
Occur for assets whose cumulative returns move in concert. When one goes up the other also goes up and vice versa. 
These positively correlated assets would not offer much or any diversification protection to each other.\n 
        """
        im = Image.open('1vs2.png')
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")
    
        with col2:
            st.image(im, caption='Trends of Assets Correlations',use_column_width='auto')
    
        with col3:
            st.write("")
            
            # Creates the title for streamlit
        st.subheader('Heatmap Showing Correlation Of Assets')
        """
Heatmap: \n
The Heat map shows the overall correlation of each asset to the other assets. Notice that the middle diagonal row is filled in with all 1’s. 
That is because they are all perfectly correlated with themselves. A value of 1 equates to perfect correlation, -1 equates to perfect negative correlation, 
and 0 equates to no correlation with values in between being relative to their distance from the extremes. A correlation value of .5 would mean 
the asset moves half as much in the same direction as the correlated asset.  A values of -0.5 would mean it moves half as much in the opposite direction 
as the correlated asset. \n
The Heat map shows the correlation coefficient or value for each asset over the entire period to each other asset. 
It also depicts the color of the intersection as darker for less correlation and lighter for more correlation, which could be either positive or negative. 
The legend on the right indicates the absolute level of correlation for each color, again positive or negative associated to each color.\n 
        """
        
        display_heat_map(choices['data'],choices['choices'])
        #display_portfolio_return(choices['combined_df'], choices['choices'])

        st.subheader('Sharpe Ratio of Assets')
        """
Sharpe Ratio: \n
The Sharpe Ratio represents the excess return of a single or many financial assets versus their
volatility or measure of risk. It is calculated using an asset’s historical returns minus the risk-free
rate divided by the standard deviation of the returns. We will offer a relevant and intuitive
definition of the ratio, including that ratios above 1 are generally considered to be “good” in the
sense that historically the asset has provided excess returns compared to the measure of risk.
Our application allows a user to see the Sharpe Ratio of any stocks selected and also the
ratio for an entire portfolio. By adding and removing or changing the investment volumes, a user
will see how the Sharpe Ratio changes based on their decisions.


        """        
        sharp_ratio_func(choices['sharpe_data'], choices['choices'])

        '''
ARIMA:\n
The primary purpose of the ARIMA model is to generate forward looking predictions of what the stock will be based purely on its historical data. We hope this 
will give the investor insight into what the stock might look like in the future if they are so inclined to invest in that particular stock.
        '''
        with st.spinner('Running Arima Model...'):
            arima_chart(choices['data'],choices['choices'])


if __name__ == "__main__":
    run()

