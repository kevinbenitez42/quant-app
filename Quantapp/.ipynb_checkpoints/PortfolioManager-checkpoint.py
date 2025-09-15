import yfinance as yf
from Quantapp.Asset import Asset
import json

class PortfolioManager:
    def __init__(self):
        self.assets = {}
    
    def add(self, ticker, direction='long'):
        self.assets[ticker] = { 'asset': Asset(ticker),
                               'direction': direction, 
                               'weight': 0.0 }
        
    def add_tickers(self,tickers):
        for ticker in tickers:
            self.add(ticker)

    def load(self, period='1y', interval='1D'):
        data = {}
        for key,value in self.assets.items():
            data[key] = value['asset'].load(period=period,
                                            interval=interval,
                                            direction=value['direction'])
        return data
    
    def load_portfolio_config(self,file_name):
        config = json.load( open( f'portfolio_configs/{file_name}.json') )
        for key,value in config.items():
            self.assets[key] = {
                'asset'     : Asset(key),
                'direction' : value['direction'],
                'weight' : value['weight']
            }

    def save_portfolio_config(self,file_name='a'):
        config = {}
        for key,value in self.assets.items():
            config[key] = { 
                'direction' : value['direction'],
                'weight' : value['weight']
            }
        json.dump( config, open( f'portfolio_configs/{file_name}.json', 'w' ) )
    
    def delete_portfolio_config(self):
        pass
    
    def retrieve_assets(self):
        return self.assets
    
    def remove(self,ticker):
        self.assets.pop(ticker)
    
    def clear(self):
        self.assets = {}
        

            