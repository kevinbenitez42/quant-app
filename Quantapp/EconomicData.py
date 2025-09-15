from fastapi import requests
import nasdaqdatalink
import pandas as pd
import yfinance as yf
nasdaqdatalink.ApiConfig.api_key = "4vgocb_MAow2x5-Xm5dv"
nasdaqdatalink.ApiConfig.proxy = "http://user:password@proxyserver:port"
import pandas_datareader as pdr
import datetime
from concurrent.futures import ThreadPoolExecutor
import requests

from quickfs import QuickFS
import os
import json

import sys
sys.path.append(r"e:\Coding Projects\Investment Analysis")

class EconomicData:
    def __init__(self):
        self.start_date = datetime.datetime(1900, 1, 1)
        self.end_date = datetime.datetime.now()  # Current date

    def get_recession_indicators(self):
        return {
            'GDP-based' : nasdaqdatalink.get('FRED/JHDUSRGDPBR'),
            'NBER based' : nasdaqdatalink.get('FRED/USREC'),
            'Recession Probabilities': nasdaqdatalink.get('FRED/RECPROUSM156N')
        }
        
    def get_inflation_data(self):
        return {
            "CPI" : nasdaqdatalink.get('FRED/CPIAUCSL'),
            "PPI" : nasdaqdatalink.get('FRED/PPIACO'),
            "Federal Funds Effective Rate": nasdaqdatalink.get('FRED/FEDFUNDS'),
            "Sticky Price Consumer Price Index": nasdaqdatalink.get("FRED/CORESTICKM159SFRBATL")
        }
    
    def get_gdp_data(self):
        return {
            'gdp' : nasdaqdatalink.get("FRED/GDP"),
            'real gdp' : nasdaqdatalink.get("FRED/GDPC1"),
            'real potential gdp': nasdaqdatalink.get("FRED/GDPPOT"),
            'gdp now': nasdaqdatalink.get("FRED/GDPNOW")
        }

    def get_bond_data(self):
        return {
            'treasury yield curve rates': nasdaqdatalink.get("USTREASURY/YIELD"),
            'treasury yield curve rates (real)': nasdaqdatalink.get("USTREASURY/REALYIELD"),
            'investment grade corporate bond yield curve rates': nasdaqdatalink.get("USTREASURY/HQMYC"),
            'US High yield Option-Adjusted Spread': nasdaqdatalink.get("FRED/BAMLH0A0HYM2"),
            'Treasury' : yf.Ticker('GOVT').history(period='max', interval='1d'),
            'Investment Corporate Bonds' : yf.Ticker('LQD').history(period='max', interval='1d'),
            'High yield Corporate Bonds' : yf.Ticker('HYG').history(period='max', interval='1d')
        }
    
    def get_housing_market_data(self):
        return {
            'Building Permits': pdr.get_data_fred("PERMIT", start=self.start_date, end=self.end_date),
            'Housing Starts': pdr.get_data_fred("HOUST", start=self.start_date, end=self.end_date),
            'New Home Sales': pdr.get_data_fred("HSN1F", start=self.start_date, end=self.end_date)
           # 'Existing Home Sales' : nasdaqdatalink.get("FRED/EXHOSLUSM495S"),
           # 'Case Shiller Home Price Index' : nasdaqdatalink.get("FRED/CSUSHPISA")
        }

    def get_leading_indicators(self):
        ism_new_order_index                   = nasdaqdatalink.get("ISM/MAN_NEWORDERS")
        average_weekly_hours_manufacturing    = nasdaqdatalink.get("FRED/PRS84006023")
        initial_claims                        = nasdaqdatalink.get("FRED/ICSA")
        manufacturers_new_orders_ex_aircraft  = nasdaqdatalink.get("FRED/AMXDNO")
        manufacturers_new_orders_consumer_goods= nasdaqdatalink.get("FRED/ACOGNO")
        leading_credit_index                   = nasdaqdatalink.get("FRED/USSLIND")
        ten_year_minus_federal_funds_rate_monthly     = nasdaqdatalink.get("FRED/T10YFFM")
     #   consumer_sentiment                            = nasdaqdatalink.get("UMICH/SOC1")
        return {
            "ism_new_order_index"                   : ism_new_order_index,
            "average_weekly_hours_manufacturing"    : pd.Series(average_weekly_hours_manufacturing['Value'],index=average_weekly_hours_manufacturing.index),
            "initial_claims"                        : pd.Series(initial_claims['Value'],index=initial_claims.index),
            "manufacturers_new_orders_ex_aircraft"  : pd.Series(manufacturers_new_orders_ex_aircraft['Value'],index=manufacturers_new_orders_ex_aircraft.index),
            "manufacturers_new_orders_consumer_goods": pd.Series(manufacturers_new_orders_consumer_goods['Value'],index=manufacturers_new_orders_consumer_goods.index),
            "leading_credit_index"                   : pd.Series(leading_credit_index['Value'],index=leading_credit_index.index),
            "ten_year_minus_federal_funds_rate_monthly"     : pd.Series(ten_year_minus_federal_funds_rate_monthly['Value'],index=ten_year_minus_federal_funds_rate_monthly.index),
         #   "consumer_sentiment"                            : pd.Series(consumer_sentiment['Index'],index=consumer_sentiment.index),
        }
    
    def get_coincident_indicators(self): 
        return {
            "non-farm payrolls": nasdaqdatalink.get("FRED/PAYEMS"),
            "aggregate_real_personal_income_ex_transfer_payments": nasdaqdatalink.get("FRED/PIECTR"),
            "industrial_production_index": nasdaqdatalink.get("FRED/INDPRO"),
            "manufacturing_and_trade_sales": nasdaqdatalink.get("FRED/M0602AUSM144SNBR")
        }
    
    def get_lagging_indicators(self):
        return {
            "average_duration_of_unemplymoyment": nasdaqdatalink.get("FRED/UEMPMEAN"),
            "inventory_sales_ratio": nasdaqdatalink.get("FRED/UEMPMEAN"),
            "change_in_unit_labor_costs":nasdaqdatalink.get("FRED/ULCNFB"),
            "average_bank_prime_lending_rate":nasdaqdatalink.get("FRED/WPRIME"),
            "commercial and industrial loans outstanding": nasdaqdatalink.get("FRED/BUSLOANS"),
            "consumer_installment_debt_to_income": nasdaqdatalink.get("FRED/TDSP"),
            "consumer_price_index_for_services": nasdaqdatalink.get("FRED/CUSR0000SAS")
        }
    
    def get_broad_market_data(self):
        return {
            'Commodities': yf.Ticker('DBC').history(period='max', interval='1d'),
            'Stock Market'     : yf.Ticker('SPY').history(period='max', interval='1d'),
            'Bonds'      : yf.Ticker('AGG').history(period='max', interval='1d'),
            'Crypto'     : yf.Ticker('BLOK').history(period='max', interval='1d'),
            'International': yf.Ticker('EFA').history(period='max', interval='1d')
        }
    
    def get_sector_data(self):
        return {
            'Healthcare' : yf.Ticker('XLV').history(period='max', interval='1d'),
            'Communications' : yf.Ticker('XLC').history(period='max', interval='1d'),
            'Technology' : yf.Ticker('XLK').history(period='max', interval='1d'),
            'Financial' : yf.Ticker('XLF').history(period='max', interval='1d'),
            'Industrial' : yf.Ticker('XLI').history(period='max', interval='1d'),
            'Materials' : yf.Ticker('XLB').history(period='max', interval='1d'),
            'Consumer Discretionary' : yf.Ticker('XLY').history(period='max', interval='1d'),
            'Consumer Staples' : yf.Ticker('XLP').history(period='max', interval='1d'),
            'Real Estate'      : yf.Ticker('XLRE').history(period='max', interval='1d'),
            'Utilities'        :yf.Ticker('XLU').history(period='max', interval='1d'),
            'Blockchain': yf.Ticker("").history(period='max',interval='1d'),
            'Energy' : yf.Ticker('XLE').history(period='max', interval='1d')
        }
        
    def get_commodity_data(self):
        return {
            'Agriculture' :  yf.Ticker('DBA').history(period='max', interval='1d'),
            'Energy'      :  yf.Ticker('DBE').history(period='max', interval='1d'),
            'Base Metals' :  yf.Ticker('DBE').history(period='max', interval='1d'),
            'Precious Metals': yf.Ticker('GLTR').history(period='max', interval='1d')
        }
    
    def get_international_data(self):    
        return {
             'Emerging Market' : yf.Ticker('EEM').history(period='max', interval='1d'),
             'Frontier Markets' : yf.Ticker('FM').history(period='max', interval='1d'),
        }
    
    def get_factor_data(self):
        return {
            'Quality': yf.Ticker('QUAL').history(period='max', interval='1d'),
            'Value' : yf.Ticker('VLUE').history(period='max', interval='1d'),
            'Momentum' : yf.Ticker('MTUM').history(period='max', interval='1d'),
            'Market Capitalization'     : yf.Ticker('SIZE').history(period='max', interval='1d'),
            'Low Beta'   : yf.Ticker('SPLV').history(period='max', interval='1d'),
            'High Beta'  : yf.Ticker('SPHB').history(period='max', interval='1d'),
            'Minimum_Volatility_Value': yf.Ticker('USMV').history(period='max', interval='1d'),
            'Minimum_Volatility_EAFE': yf.Ticker('EFAV').history(period='max', interval='1d'),
            'Minimum_Volatility_Emerging_Markets': yf.Ticker('EEMV').history(period='max', interval='1d'),
            'Fixed_Income_investment_grade' : yf.Ticker('IGEB').history(period='max', interval='1d'),
            'Fixed_Income_Balanced_Risk' : yf.Ticker('FIBR').history(period='max', interval='1d'),
            'Fixed_Income_High_yield' : yf.Ticker('HYDB').history(period='max', interval='1d'),
            'Multi_factor_usa' : yf.Ticker('FIBR').history(period='max', interval='1d'),
            'Multi_factor_international' : yf.Ticker('INTF').history(period='max', interval='1d'),  
            'Multi_factor_global' : yf.Ticker('ACWF').history(period='max', interval='1d'),
        }
        
    def get_allocation_data(self):
        return {
            'Growth' : yf.Ticker('AOR').history(period='max', interval='1d'),
            'Moderate' : yf.Ticker('AOM').history(period='max', interval='1d'),
            'Aggresive': yf.Ticker('AOA').history(period='max', interval='1d'),
            'Conservative': yf.Ticker('AOK').history(period='max', interval='1d'),
        }
        
    def get_volatility_data(self):
        
        return {
            'VIX (1 Month)': yf.Ticker('VIX').history(period='max', interval='1d'),
            'VIX (6 Month)': yf.Ticker('VIXM').history(period='max', interval='1d'),
            'SKEW': yf.Ticker('^SKEW').history(period='max', interval='1d'),
            'MOVE': yf.Ticker('^MOVE').history(period='max', interval='1d'),
        }
    
    def get_strategy_data(self):
        return {
            'Active Investing': yf.Ticker('QAI').history(period='max', interval='1d'),
            'Beta Rotation': yf.Ticker('BTAL').history(period='max', interval='1d'),
            'Covered Calls': yf.Ticker('PBP').history(period='max', interval='1d'),
            'Hedged'       : yf.Ticker('PHDG').history(period='max', interval='1d')
        }
            
    def get_market_assets(self):
        # Get the market tables
        tables = self.retrieve_market_tables()

        sp500_table = tables["SP500_TABLE"]
    #    qqq_table = tables["NASDAQ_100_TABLE"]
        dia_table = tables["DIA_TABLE"]
        #russell_1000_table = tables["Russell_1000_TABLE"]

        # Retrieve all companies from each sector
        xlk_table = sp500_table[sp500_table['Sector'] == 'Information Technology']
        xlf_table = sp500_table[sp500_table['Sector'] == 'Financials']
        xlv_table = sp500_table[sp500_table['Sector'] == 'Health Care']
        xli_table = sp500_table[sp500_table['Sector'] == 'Industrials']
        xly_table = sp500_table[sp500_table['Sector'] == 'Consumer Discretionary']
        xle_table = sp500_table[sp500_table['Sector'] == 'Energy']
        xlb_table = sp500_table[sp500_table['Sector'] == 'Materials']
        xlc_table = sp500_table[sp500_table['Sector'] == 'Communication Services']
        xlre_table = sp500_table[sp500_table['Sector'] == 'Real Estate']
        xlp_table = sp500_table[sp500_table['Sector'] == 'Consumer Staples']
        xlu_table = sp500_table[sp500_table['Sector'] == 'Utilities']

        market_assets = {
            "INDICES": ['SPY', 'QQQ', 'DIA', 'IWM'],
            "SECTORS": ['XLF', 'XLK', 'XLV', 'XLC', 'XLI', 'XLU', 'XLB', 'VNQ', 'XLP', 'XLY', 'XBI', 'XLE'],
            "INDUSTRIES": ['SPY', 'SMH', 'KRE', 'KIE', 'KBE'],
            "SPY_HOLDINGS": sp500_table['Symbol'].tolist(),
        #    "QQQ_HOLDINGS": qqq_table['Symbol'].tolist(),
            "DIA_HOLDINGS": dia_table['Symbol'].tolist(),
            # "RUSSELL_1000_HOLDINGS": russell_1000_table['Symbol'].tolist(),
            "XLK_HOLDINGS": xlk_table['Symbol'].tolist(),
            "XLF_HOLDINGS": xlf_table['Symbol'].tolist(),
            "XLI_HOLDINGS": xli_table['Symbol'].tolist(),
            "XLV_HOLDINGS": xlv_table['Symbol'].tolist(),
            "XLU_HOLDINGS": xlu_table['Symbol'].tolist(),
            "XLB_HOLDINGS": xlb_table['Symbol'].tolist(),
            "XLY_HOLDINGS": xly_table['Symbol'].tolist(),
            "XLRE_HOLDINGS": xlre_table['Symbol'].tolist(),
            "XLC_HOLDINGS": xlc_table['Symbol'].tolist(),
            "XLE_HOLDINGS": xle_table['Symbol'].tolist(),
            "XLP_HOLDINGS": xlp_table['Symbol'].tolist(),
            "BONDS": ['AGG', 'IEF', 'TLT', 'HYG', 'LQD', 'BKLN'],# 'TIPS' EXCLUDED
            "AGRICULTURE": ['DBA', 'CORN', 'WEAT', 'SOYB'],
            "PRECIOUS_METALS": ['GLD', 'SLV', 'GDX', 'XME','NUGT','GDJ'],
            "CRYPTO": ['BTC-USD', 'ETH-USD', 'LTC-USD', 'ADA-USD', 'SOL-USD'],
            "ENERGY": ['USO', 'UNG', 'OIH', 'XOP', 'TAN', 'ICLN', 'URA', 'URNM', 'GUSH', 'KOLD'],
            "FOREIGN_MARKETS": [
                'EWZ', 'EWJ', 'EWA', 'EWG', 'EWW', 'EEM', 'EFA', 'FEZ', 'INDA', 'EWU',
                'EWG', 'EWL', 'XIU', 'VAS', 'ENZL', 'TUR', 'EZA', 'EWS', 'EWH'
            ],
            "PRIMARY_SECTORS": ['USO', 'GLD', 'SPY', 'VNQ', 'GBTC', 'EFA', 'TLT', 'TIP','UUP'],
            "MAJOR_CURRENCY_PAIRS": ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'CHF=X', 'CAD=X', 'AUDUSD=X', 'NZDUSD=X'],
            "MINOR_CURRENCY_PAIRS": ['EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'GBPCHF=X', 'AUDNZD=X'],
            "EXOTIC_CURRENCY_PAIRS": ['TRY=X', 'ZAR=X', 'SGD=X', 'HKD=X', 'MXN=X'],
            "CROSS_CURRENCY_PAIRS": ['EURCHF=X', 'EURAUD=X', 'GBPAUD=X', 'CHFJPY=X'],
            "CAPITALIZATIONS": ['SPY', 'IJH', 'IJR'],
            "INNOVATION": ['ARKG', 'ARKF', 'ARKK'],
            "LONG_LEVERAGE": ['TQQQ', 'SOXL', 'SPXL', 'TNA', 'BOIL', 'NUGT', 'ERX', 'DPST'],
            "SHORT_LEVERAGE": ['SQQQ', 'SPXS', 'UDOW', 'SSO', 'TECL', 'FAS', 'NVDA', 'TQQQ', 'VXX', 'UVXY', 'VIXY', 'UVIX', 'SVXY', 'SOXS', 'TZA', 'USD', 'TSLL', 'LABU', 'DPST', 'NUGT', 'CONL'],
            "SINGLE_FACTOR": ['QUAL', 'VLUE', 'MTUM', 'SIZE', 'USMV'],
            "MULTI_FACTOR": ['LRGF', 'INTF', 'GLOF'],
            "MINIMUM_VOLATILITY": ['USMV', 'EFAV', 'EEMV'],
        }

        return market_assets
    '''
    def retrieve_market_tables(self):
        # URLs for the market data
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        dow_url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        nasdaq_url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        russell_1000_url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'

        # Retrieve S&P 500 data
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_table = sp500_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        sp500_table = sp500_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})

        # Retrieve NASDAQ 100 data

        
        qqq_table = pd.read_html(nasdaq_url)[4]
        qqq_table = qqq_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        qqq_table = qqq_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})
        qqq_table = pd.merge(qqq_table, sp500_table[['Symbol', 'Sub-Industry']], on='Symbol', how='left')
        qqq_table['Sub-Industry'] = qqq_table['Sub-Industry_x'].combine_first(qqq_table['Sub-Industry_y'])
        qqq_table = qqq_table.drop(columns=['Sub-Industry_x', 'Sub-Industry_y'])
        
        # Retrieve Dow Jones Industrial Average data
        tables = pd.read_html(dow_url)
        dia_table = tables[2]
        dia_table = dia_table[['Symbol', 'Industry']]
        dia_table = pd.merge(dia_table, sp500_table[['Symbol', 'Sector']], on='Symbol', how='left')
        dia_table = pd.merge(dia_table, sp500_table[['Symbol', 'Sub-Industry']], on='Symbol', how='left')
        dia_table = dia_table.drop(columns=['Industry'])

        # Retrieve Russell 1000 data
        # russell_1000_table = pd.read_html(russell_1000_url)[2]
        # russell_1000_table = russell_1000_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        # russell_1000_table = russell_1000_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})

        # Compile all tables into a dictionary
        data_dict = {
            "SP500_TABLE": sp500_table,
            "NASDAQ_100_TABLE": qqq_table,
            "DIA_TABLE": dia_table,
            # "Russell_1000_TABLE": russell_1000_table
        }

        return data_dict
    '''
    
    def retrieve_market_tables(self):
        # URLs for the market data
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        dow_url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        #nasdaq_url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        russell_1000_url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(sp500_url, headers=headers)
        # Retrieve S&P 500 data
        sp500_table = pd.read_html(response.text)[0]
        sp500_table = sp500_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        sp500_table = sp500_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})

        # Retrieve NASDAQ 100 data

        
        #qqq_table = pd.read_html(nasdaq_url)[4]
        #qqq_table = qqq_table[['Ticker', 'GICS Sector', 'GICS Sub-Industry']]
        #qqq_table = qqq_table.rename(columns={'Ticker': 'Symbol', 'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})
        #qqq_table = pd.merge(qqq_table, sp500_table[['Symbol', 'Sub-Industry']], on='Symbol', how='left')
        #qqq_table['Sub-Industry'] = qqq_table['Sub-Industry_x'].combine_first(qqq_table['Sub-Industry_y'])
        #qqq_table = qqq_table.drop(columns=['Sub-Industry_x', 'Sub-Industry_y'])
        
        # Retrieve Dow Jones Industrial Average data
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(dow_url, headers=headers)
        tables = pd.read_html(response.text)
        dia_table = tables[2]
        dia_table = dia_table[['Symbol', 'Industry']]
        dia_table = pd.merge(dia_table, sp500_table[['Symbol', 'Sector']], on='Symbol', how='left')
        dia_table = pd.merge(dia_table, sp500_table[['Symbol', 'Sub-Industry']], on='Symbol', how='left')
        dia_table = dia_table.drop(columns=['Industry'])

        # Retrieve Russell 1000 data
        # russell_1000_table = pd.read_html(russell_1000_url)[2]
        # russell_1000_table = russell_1000_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        # russell_1000_table = russell_1000_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})

        # Compile all tables into a dictionary
        data_dict = {
            "SP500_TABLE": sp500_table,
        #    "NASDAQ_100_TABLE": qqq_table,
            "DIA_TABLE": dia_table,
            # "Russell_1000_TABLE": russell_1000_table
        }

        return data_dict
    
    def retrieve_market_data(self):
        # Get the market tables
        tables = self.retrieve_market_tables()

        sp500_table = tables["SP500_TABLE"]
    #    qqq_table = tables["NASDAQ_100_TABLE"]
        dia_table = tables["DIA_TABLE"]
        #russell_1000_table = tables["Russell_1000_TABLE"]

        # Retrieve all companies from each sector and store in dictionary
        data_dict = {
            "SP500": sp500_table,
        #    "NASDAQ_100": qqq_table,
            "DIA": dia_table,
            # "Russell_1000": russell_1000_table,
            "Information Technology": sp500_table[sp500_table['Sector'] == 'Information Technology'],
            "Financials": sp500_table[sp500_table['Sector'] == 'Financials'],
            "Health Care": sp500_table[sp500_table['Sector'] == 'Health Care'],
            "Industrials": sp500_table[sp500_table['Sector'] == 'Industrials'],
            "Consumer Discretionary": sp500_table[sp500_table['Sector'] == 'Consumer Discretionary'],
            "Energy": sp500_table[sp500_table['Sector'] == 'Energy'],
            "Materials": sp500_table[sp500_table['Sector'] == 'Materials'],
            "Communication Services": sp500_table[sp500_table['Sector'] == 'Communication Services'],
            "Real Estate": sp500_table[sp500_table['Sector'] == 'Real Estate'],
            "Consumer Staples": sp500_table[sp500_table['Sector'] == 'Consumer Staples'],
            "Utilities": sp500_table[sp500_table['Sector'] == 'Utilities']
        }

        return data_dict
    
    def generate_series(self,tickers, columns=['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], period='10y', interval='1d'):
        """
        Generate a DataFrame or Series containing the specified columns for the given tickers.

        Parameters:
        - tickers: List of ticker symbols or a single ticker symbol.
        - columns: List of columns to retrieve or a single column to retrieve (default is ['Close']).
        - period: Data period to retrieve (default is '1y').
        - interval: Data interval to retrieve (default is '1d').

        Returns:
        - pd.DataFrame or pd.Series with the specified columns for the given tickers.
        """
        # Ensure tickers and columns are lists
        if isinstance(tickers, str):
            tickers = [tickers]
        if isinstance(columns, str):
            columns = [columns]

        tickers = [ticker.replace('.', '-') for ticker in tickers]
        try:
            df = yf.download(tickers, period=period, interval=interval, progress=False)
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            return pd.DataFrame()
        
        # Check if the specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns.get_level_values(0)]
        if missing_columns:
            print(f"Error: The following columns are not available: {missing_columns}")
            print(f"Possible columns are: {df.columns.get_level_values(0).unique().tolist()}")
            return pd.DataFrame()
        
        df = df[columns]
        
        # Handle the case where there is only one ticker and one column
        if len(tickers) == 1 and len(columns) == 1:
            return df[columns[0]].rename(tickers[0].replace('-', '.'))
        
        # Handle the case where there is only one ticker
        if len(tickers) == 1:
            df.columns = [col.replace('-', '.') for col in df.columns]
        else:
            # If only one column is selected, return a DataFrame with tickers as column names
            if len(columns) == 1:
                df = df[columns[0]]
                df.columns = [col.replace('-', '.') for col in df.columns]
            else:
                # Flatten the multi-level columns if multiple tickers are requested
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = pd.MultiIndex.from_tuples([(col[1], col[0]) for col in df.columns.values])
                else:
                    df.columns = pd.MultiIndex.from_tuples([(col.split('.')[0], col.split('.')[1]) for col in df.columns])
        
        return df
    
    def get_sector_info(ticker):
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', 'N/A')
            sub_industry = stock.info.get('industry', 'N/A')
            return {'Ticker': ticker, 'Sector': sector, 'Sub-Industry': sub_industry}
        except Exception as e:
            #print(f"Error fetching data for {ticker}: {e}")
            return {'Ticker': ticker, 'Sector': 'N/A', 'Sub-Industry': 'N/A'}

    def fetch_ticker_info(self, ticker):
        info = self.get_sector_info(ticker)
        print(info)
        print(yf.Ticker(ticker).info)
        #market_cap = yf.Ticker(ticker).info.get('marketCap')
        #return info['Sector'], info['Sub-Industry'], market_cap

    def get_market_caps(self,table):
        #print("Starting market cap retrieval process...")
        
        tickers = table['Symbol'].tolist()
        #print(f"Original tickers: {tickers[:10]}...")  # Print first 10 for brevity

        # Optimize ticker adjustment
        tickers = ['BRK-B' if symbol == 'BRK.B' else 'BF-B' if symbol == 'BF.B' else symbol for symbol in tickers]
        #print(f"Adjusted tickers: {tickers[:10]}...")  # Print first 10 for brevity

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.fetch_ticker_info, tickers))

        # Unpack results
        sectors, sub_industries, market_caps = zip(*results)

        table['Sector'] = sectors
        table['Sub-Industry'] = sub_industries
        table['Market Cap'] = market_caps
        
        #print("Market cap retrieval process completed.")
        return table
    
    def get_market_cap_threshold_companies(self,info):
        """
        Calculates market cap rankings and identifies companies contributing to specified cumulative market cap thresholds.

        Parameters:
            info (pd.DataFrame): DataFrame containing at least 'Symbol' and 'Market Cap' columns.

        Returns:
            dict: A dictionary where keys are threshold labels (e.g., 'Top 50%') and values are lists of company dictionaries
                containing 'Symbol', 'Market Cap', 'Market Cap %', 'Cumulative Market Cap %', and 'Rank'.
        """
        # Step 1: Create a DataFrame of market caps
        market_caps = pd.DataFrame(info[['Symbol', 'Market Cap']])
        
        # Step 2: Sort companies by market cap in descending order
        market_caps = market_caps.sort_values(by='Market Cap', ascending=False).reset_index(drop=True)

        # Step 3: Calculate total market cap
        total_market_cap = market_caps['Market Cap'].sum()

        # Step 4: Calculate individual Market Cap %
        market_caps['Market Cap %'] = (market_caps['Market Cap'] / total_market_cap) * 100

        # Step 5: Calculate cumulative Market Cap %
        market_caps['Cumulative Market Cap %'] = market_caps['Market Cap %'].cumsum()

        # Step 6: Assign Rank
        market_caps['Rank'] = market_caps.index + 1

        # Step 7: Define thresholds
        thresholds = [50, 80]  # You can adjust or add more thresholds as needed
        threshold_dict = {}

        for threshold in thresholds:
            # Find the first index where cumulative market cap meets or exceeds the threshold
            idx = market_caps[market_caps['Cumulative Market Cap %'] >= threshold].index[0]

            # Select companies up to that index
            companies = market_caps.loc[:idx, ['Symbol', 'Market Cap', 'Market Cap %', 'Cumulative Market Cap %', 'Rank']]

            # Convert to list of dictionaries
            companies_list = companies.to_dict('records')

            # Add to the threshold dictionary with appropriate key
            threshold_key = f'Top {threshold}%'
            threshold_dict[threshold_key] = companies_list

        return threshold_dict


class CompanyData:
    
    def __init__(self, ticker_str, client=None,save_path=None):
        """
        Initializes the CompanyData class with a ticker and an optional QuickFS client.
        
        Parameters:
            ticker (str): The stock ticker symbol for the company.
            client (QuickFSClient, optional): An instance of QuickFSClient to interact with the QuickFS API.
        """
        self.ticker_str = ticker_str
        self.client = client if client else QuickFS()
        if save_path is None:
            self.save_path = os.path.join(os.getcwd(), 'data', 'company_data')
        else:
            self.save_path = save_path
        
    def get_metrics(self):
        """
        Retrieves the metrics available from the QuickFS API.
        
        Parameters:
            client (QuickFSClient): The client object used to interact with the API.
        
        Returns:
            list: A list of dictionaries containing the available metrics and their descriptions.
        """
        # Retrieve the metrics available from the API
        metrics = self.client.get_available_metrics()
        
        # Return the list of metrics
        return metrics
    
    def get_latest_earnings_date(self):
        """
        Retrieves the most recent earnings date for the company.
        
        Returns:
            str: The most recent earnings date in 'YYYY-MM-DD' format or None if no dates found.
        """
        earnings_dates = yf.Ticker(self.ticker_str).earnings_dates.dropna().index
        if len(earnings_dates) > 0:
            last_earnings_date = str(earnings_dates[0].strftime('%Y-%m-%d'))
            return last_earnings_date
        return None
    
    def process_full_data(self, full_data, client):
        """
        Processes the full_data dictionary to extract company metadata and financials.
        It converts financial data (annual and quarterly) into DataFrames with sorted columns,
        and then separates them into dictionaries based on statement types using available metrics.
        
        Parameters:
            full_data (dict): Dictionary containing keys 'metadata' and 'financials' (with keys 'annual' and 'quarterly').
            client: Parameter used to retrieve metrics via get_metrics(client).
        
        Returns:
            dict: A dictionary containing:
                'ticker': company ticker,
                'metadata': company metadata,
                'financials_annual': dictionary of annual financial DataFrames,
                'financials_quarterly': dictionary of quarterly financial DataFrames.
        """
        import pandas as pd

        def create_financials_dict(financials_df, client):
            # Retrieve the metrics from the client using get_metrics
            
            metrics = self.get_metrics()
            # Identify metrics that are not present in the DataFrame
            metrics_not_in_df = []
            for field in metrics:
                if field['metric'] not in financials_df.columns:
                    metrics_not_in_df.append(field['metric'])

            # Filter out metrics not in the DataFrame and sort the remaining metrics
            metrics = [field for field in metrics if field['metric'] not in metrics_not_in_df]
            metrics = sorted(metrics, key=lambda x: x['metric'])
            

            
            
            # Extract metrics based on statement type
            metrics_income_statement    = [field['metric'] for field in metrics if field['statement_type'] == 'income_statement']
            metrics_balance_sheet       = [field['metric'] for field in metrics if field['statement_type'] == 'balance_sheet']
            metrics_cash_flow_statement = [field['metric'] for field in metrics if field['statement_type'] == 'cash_flow_statement']
            metrics_computed             = [field['metric'] for field in metrics if field['statement_type'] == 'computed']
            metrics_misc                = [field['metric'] for field in metrics if field['statement_type'] == 'misc']
    
            # Create separate DataFrames for each statement type by concatenating metrics_misc with the respective group
            income_statement_df     = pd.concat([financials_df[metrics_misc], financials_df[metrics_income_statement]], axis=1)
            balance_sheet_df        = pd.concat([financials_df[metrics_misc], financials_df[metrics_balance_sheet]], axis=1)
            cash_flow_statement_df  = pd.concat([financials_df[metrics_misc], financials_df[metrics_cash_flow_statement]], axis=1)
            computed_df              = pd.concat([financials_df[metrics_misc], financials_df[metrics_computed]], axis=1)
            misc_df                 = financials_df[metrics_misc]
            
            #fiscal_year_key
            #fiscal_year_number
            #fiscal_quarter_key
            #fiscal_quarter_number 
            
            #check if fiscal_year_key, fiscal_year_number, fiscal_quarter_key, fiscal_quarter_number are in the financials_df
            #if they are, add columns to each dataframe
            
            if 'fiscal_year_key' in financials_df.columns:
                income_statement_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                balance_sheet_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                cash_flow_statement_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                computed_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                misc_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                
            if 'fiscal_year_number' in financials_df.columns:
                income_statement_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                balance_sheet_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                cash_flow_statement_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                computed_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                misc_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                
            if 'fiscal_quarter_key' in financials_df.columns:
                income_statement_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                balance_sheet_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                cash_flow_statement_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                computed_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                misc_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                
            if 'fiscal_quarter_number' in financials_df.columns:
                income_statement_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                balance_sheet_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                cash_flow_statement_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                computed_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                misc_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                
            # Return a dictionary with the separated DataFrames
            return {
                'income_statement': income_statement_df,
                'balance_sheet': balance_sheet_df,
                'cash_flow_statement': cash_flow_statement_df,
                'computed': computed_df,
                'misc': misc_df
            }
        
        # Get ticker; default to 'AAPL' if not provided in full_data
        ticker = full_data['metadata']['symbol']
        
        # Extract metadata and financials from full_data
        company_metadata     = full_data['metadata']
        financials_annual    = full_data['financials']['annual']
        financials_quarterly = full_data['financials']['quarterly']
        
        # Create DataFrames for annual and quarterly financials with sorted columns
        financials_annual_df = pd.DataFrame(financials_annual).reindex(
            sorted(pd.DataFrame(financials_annual).columns), axis=1
        )
        financials_quarterly_df = pd.DataFrame(financials_quarterly).reindex(
            sorted(pd.DataFrame(financials_quarterly).columns), axis=1
        )
        

        
        # Create dictionaries for annual and quarterly financials
        financials_annual_dict    = create_financials_dict(financials_annual_df, self.client)
        financials_quarterly_dict = create_financials_dict(financials_quarterly_df, self.client)
        
        return {
            'ticker': ticker,
            'metadata': company_metadata,
            'financials_annual': financials_annual_dict,
            'financials_quarterly': financials_quarterly_dict
        }
             
    def retrieve_data_from_API(self):
        print('Querying data API!')
        #convert dictionary to 
        data = self.client.get_data_full(symbol=self.ticker_str)
        data = self.process_full_data(data, self.client)
        print('Data retreived successfully!\n')
        self.save_company_info(self.ticker_str, data['financials_annual'], data['financials_quarterly'], data['metadata'])
        print('Data saved successfully!\n')
        return data
    
    def retrieve_data(self, data_type='annual', statement_type='income_statement', should_update=False):
        # Check if the ticker folder exists; if not, create it
        ticker_folder = os.path.join(self.save_path, 'company_data', self.ticker_str)
        if not os.path.exists(ticker_folder):
            os.makedirs(ticker_folder)
            print('Company folder does not exist, creating it now!\n')
        else:
            print('Company folder exists\n')

        # Check if the company folder is empty
        if len(os.listdir(ticker_folder)) == 0:
            print(ticker_folder)
            print('No data found for this company\n')
            self.retrieve_data_from_API()
        else:
            print('Data found for this company\n')
            
        if should_update:
            print('Manually updating data\n')
            self.retrieve_data_from_API()
        else:
            print('Set should_update to True to manually update data\n')
            
        # Check if the data is up to date; if not, retrieve it from the API
        # (This comment is a reminder for future implementation if needed.)

        # If the data exists and is up to date, load it from disk according to the fiscal period
        if data_type == 'annual':
            print('Retrieving annual data\n')
            return pd.read_csv(os.path.join(ticker_folder, data_type, statement_type + '.csv'))
        elif data_type == 'quarterly':
            print('Retrieving quarterly data\n')
            return pd.read_csv(os.path.join(ticker_folder, data_type, statement_type + '.csv'))
        elif data_type == 'metadata':
            print('Retrieving metadata\n')
            with open(os.path.join(ticker_folder, 'metadata.json')) as f:
                return json.load(f)
        else:
            print('Invalid fiscal period\n')

    def save_company_info(self,ticker, financials_data_annual, financials_data_quarterly, company_metadata):
        """
        Saves company financial data (annual and quarterly) and metadata to disk.
        
        This function:
        - Creates a folder for the given ticker under 'company_data' if it doesn't exist.
        - For each period ('annual' and 'quarterly'), creates a subfolder and saves each
            financial DataFrame as a CSV file.
        - Saves the company metadata as a JSON file in the ticker folder.
        
        Parameters:
        ticker (str): The company ticker.
        financials_data_annual (dict): Dictionary of annual financial DataFrames.
        financials_data_quarterly (dict): Dictionary of quarterly financial DataFrames.
        company_metadata (dict): Dictionary containing company metadata.
        """
        import os, json
        save_path = self.save_path
        parent_folder = os.path.join(save_path, 'company_data')
        
        #parent_folder = 'company_data'
        ticker_folder = os.path.join(parent_folder, ticker)
        if not os.path.exists(ticker_folder):
            os.makedirs(ticker_folder)
            print(f"Created folder: {ticker_folder}")

        # Function to save financial data for a given period
        def save_financials_for_period(financials_data, period):
            period_folder = os.path.join(ticker_folder, period)
            if not os.path.exists(period_folder):
                os.makedirs(period_folder)
                print(f"Created folder for {period} data: {period_folder}")

            for key, df in financials_data.items():
                file_path = os.path.join(period_folder, f'{key}.csv')
                df.to_csv(file_path)
                print(f"{key} data saved to {file_path}")

        # Save annual financials
        save_financials_for_period(financials_data_annual, 'annual')

        # Save quarterly financials
        save_financials_for_period(financials_data_quarterly, 'quarterly')

        # Save company metadata to JSON
        metadata_path = os.path.join(ticker_folder, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(company_metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")