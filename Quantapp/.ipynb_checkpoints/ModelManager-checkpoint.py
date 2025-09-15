import pmdarima as pm
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class ModelManager:
    
    def __init__(self):
        pass
    

    def create_arima_model(self,training_set, start_p=1, start_q=1, max_p=3,max_q=3,start_P=0, d=1,D=1,trace=True, seasonal=14):
        return pm.auto_arima(training_set, start_p=start_p, start_q=start_q,
                             max_p=max_p, max_q=max_q, m=14,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',  
                             suppress_warnings=True,  
                             stepwise=True)  
    
    def create_garch_model(self,vol="GARCH",p=1,o=0,q=1,dist="Normal"):
        am =  arch_model(returns, vol="GARCH", p=1, o=0, q=1, dist="Normal")
        return am.fit(update_freq=5, disp="off")


class FeatureGenerator:
    
    def __init__(self):
        self.sc = StandardScaler()
        
    
    def train_test_split(self, series, percent_split):
        X = series.values
        size = int(len(X) * percent_split)
        y_train,y_test =X [0: size], X[size:len(X)]
        
    def fill_missing_dates(self, series):
        index = series.index
        date_index = pd.date_range(start=index[0], end=index[-1], freq='D')
        series = series.reindex(date_index)
        series = series.fillna(method='ffill')
        return series
    
    def fill_missing_dates_df(self,data_frame):
        data_frame_ = pd.DataFrame()
        for column in data_frame.columns:
            data_frame_[column] = self.fill_missing_dates(data_frame[column])
        return data_frame_

    def reduce_dimensionality(self, df, threshold):
        pca        = PCA(n_components=len(df.columns))
        scaled_df  = pd.DataFrame(self.sc.fit_transform(df))
        components = pd.DataFrame(pca.fit_transform(scaled_df.values))
        
        for i in range(len(df)):
            ev = np.cumsum(pca.explained_variance_ratio_*100)[i]
            if ev >= (threshold*100):
                components = pd.DataFrame(components.values)
                return components.iloc[:, :i]
            
    def monthly_to_daily(self,data):
        dates = pd.date_range(data.index[0], data.index[-1], freq='D')
        s_daily = data.reindex(dates, method='ffill')
        return s_daily.fillna(0)
    
    def calculate_differences(self,df):
        diff_ = pd.DataFrame()
        for column in df.columns:
            for column2 in df.columns:
                diff_[f'{column} min {column2}'] = df[column] - df[column2]
        return diff_
            
