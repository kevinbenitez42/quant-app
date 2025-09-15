import nasdaqdatalink
import pandas as pd
import yfinance as yf
nasdaqdatalink.ApiConfig.api_key = "D2S-LBYJrbd-1mNzGRGg"

class DataManager:
    def __init__(self):
        pass
    
class EconomicData:
    def __init__(self):
        pass
    
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
            'real_potential_gdp': nasdaqdatalink.get("FRED/GDPPOT"),
            'gdp_now': nasdaqdatalink.get("FRED/GDPNOW")
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

    def get_leading_indicators(self):
        ism_new_order_index                   = nasdaqdatalink.get("ISM/MAN_NEWORDERS")
        average_weekly_hours_manufacturing    = nasdaqdatalink.get("FRED/PRS84006023")
        initial_claims                        = nasdaqdatalink.get("FRED/ICSA")
        manufacturers_new_orders_ex_aircraft  = nasdaqdatalink.get("FRED/AMXDNO")
        manufacturers_new_orders_consumer_goods= nasdaqdatalink.get("FRED/ACOGNO")
        Building_permits_new_private_housing   = nasdaqdatalink.get("FRED/PERMIT")
        leading_credit_index                   = nasdaqdatalink.get("FRED/USSLIND")
        ten_year_minus_federal_funds_rate_monthly     = nasdaqdatalink.get("FRED/T10YFFM")
        consumer_sentiment                            = nasdaqdatalink.get("UMICH/SOC1")
        return {
            "ism_new_order_index"                   : ism_new_order_index,
            "average_weekly_hours_manufacturing"    : pd.Series(average_weekly_hours_manufacturing['Value'],index=average_weekly_hours_manufacturing.index),
            "initial_claims"                        : pd.Series(initial_claims['Value'],index=initial_claims.index),
            "manufacturers_new_orders_ex_aircraft"  : pd.Series(manufacturers_new_orders_ex_aircraft['Value'],index=manufacturers_new_orders_ex_aircraft.index),
            "manufacturers_new_orders_consumer_goods": pd.Series(manufacturers_new_orders_consumer_goods['Value'],index=manufacturers_new_orders_consumer_goods.index),
            "Building_permits_new_private_housing"   : pd.Series(Building_permits_new_private_housing['Value'] ,index=Building_permits_new_private_housing.index),
            "leading_credit_index"                   : pd.Series(leading_credit_index['Value'],index=leading_credit_index.index),
            "ten_year_minus_federal_funds_rate_monthly"     : pd.Series(ten_year_minus_federal_funds_rate_monthly['Value'],index=ten_year_minus_federal_funds_rate_monthly.index),
            "consumer_sentiment"                            : pd.Series(consumer_sentiment['Index'],index=consumer_sentiment.index),
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
            'VIX (6 Month)': yf.Ticker('VIXM').history(period='max', interval='1d')
        }
    
    def get_strategy_data(self):
        return {
            'Active Investing': yf.Ticker('QAI').history(period='max', interval='1d'),
            'Beta Rotation': yf.Ticker('BTAL').history(period='max', interval='1d'),
            'Covered Calls': yf.Ticker('PBP').history(period='max', interval='1d'),
            'Hedged'       : yf.Ticker('PHDG').history(period='max', interval='1d')
        }