import pandas as pd
import yfinance as yf
import os 

class UniverseManager:
    def __init__(self, file_path='csv_files/S&P 500.csv'):
        self.set_file_path(file_path)
        self.load_csv()
        
    def load_csv(self):
        self.data = pd.read_csv(self.file_path)
        self.clean_market_cap()
    
    def set_file_path(self,file_path):
        self.file_path = os.path.abspath(file_path)
        
    def clean_market_cap(self):
        self.data['Market Cap'] = self.data['Market Cap'].apply(lambda x: int(x
                            .replace('M','000000')
                            .replace(',','')
                            .replace(' ','')))
        
    def retrieve_assets(self, Sector=None, Industry=None, SubIndustry=None):
        df = self.data
        if Sector != None:
            df = df[self.data['Sector'] == Sector]
        if Industry != None:
            df = df[self.data['Industry'] == Industry]
        if SubIndustry != None:
            df = df[self.data['Sub-Industry'] == SubIndustry]
 
        return df
    
    def retrieve_unique_asset_classes(self, Sector=None, Industry=None):
        df = self.data
        if Sector != None:
            df = df[self.data['Sector'] == Sector]
            return df['Industry'].unique()
        if Industry != None:
            df = df[self.data['Industry'] == Industry]
            return df['SubIndustry'].unique()
        return df
    
    def create_market_index(self, Sector=None, Industry=None):
        asset_list = []
        market_cap_weighted_prices = {}
        
        unique_asset_classes = self.retrieve_unique_asset_classes(Sector=Sector,Industry=Industry)
        companies            = self.retrieve_assets(Sector=Sector,Industry=Industry)
        total_market_cap = companies['Market Cap'].sum()
        companies['Weight(Market Cap)'] = companies['Market Cap'].apply(lambda mc: mc / total_market_cap)
        weights = companies.copy()[['Symbol','Weight(Market Cap)']]

        for symbol in companies['Symbol']:
            prices = yf.Ticker(symbol).history(period='max',interval='1D')['Close']
            weight = weights[weights['Symbol'] == symbol]['Weight(Market Cap)'].iloc[0]
            market_cap_weighted_price = prices.apply(lambda x: weight * x) 
            market_cap_weighted_prices[symbol] = market_cap_weighted_price
            
        return pd.DataFrame(market_cap_weighted_prices).sum(axis=1)
    

    def retrieve_universe(self):
        return self.data
        
    
        
    