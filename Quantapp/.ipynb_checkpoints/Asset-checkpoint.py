from Quantapp.UniverseManager import UniverseManager
import yfinance as yf

class Asset:
    def __init__(self, ticker,period='1y', interval='1D'):
        self.um = UniverseManager(file_path='csv_files/S&P 500.csv')
        universe = self.um.retrieve_universe()
        self.ticker = ticker
        asset = universe.loc[universe['Symbol'] == self.ticker]
        self.data ={
            'period'       : period,
            'interval'     : interval,
            'Symbol'       : asset['Symbol'].tolist()[0],
            'Industry'     : asset['Industry'].tolist()[0],
            'Sub-Industry' : asset['Sub-Industry'].tolist()[0],
            'Market Cap'   : asset['Market Cap'].tolist()[0],
            'Description'  : asset['Description'].tolist()[0],
            'Company Name' : asset['Company Name'].tolist()[0],
            'Direction'    : None,
            'Ticker Data'  : None       
        }
        
    def load(self, period='1y', interval='1D', direction='long'):
        data_ = self.data
        data_['Ticker Data'] = yf.Ticker(self.ticker).history(period=period,
                                                             interval=interval)
        
        if direction == 'long':
            return data_
        elif direction =='short':
            data_['Ticker Data'] = -data_['Ticker Data']
            return data_
        else:
            return None

        


        
    