# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:58:37 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   Preprocess import Preprocess

class Signal(Preprocess):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.signal_path = os.path.join(self.data_path, "signal")
        if os.path.exists(self.signal_path) == False: os.makedirs(self.signal_path)
        
        self.trend_window  = 5
        self.zscore_window = 10
        
    def _get_trend(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        
        df_out = (df.sort_values(
            "date").
            assign(
                window    = self.trend_window,
                roll_mean = lambda x: x.value.ewm(span = self.window, adjust = False).mean(),
                signal    = lambda x: x.value - x.roll_mean).
            drop(columns = ["roll_mean"]))
        
        return df_out
    
    def _lag_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_out = (df.sort_values(
            "date").
            assign(lag_spread = lambda x: x.spread.shift()))
        
        return df_out
        
    def get_trend(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "CreditTrend.parquet")
        
        try:
            
            if verbose == True: print("Trying to find trend data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")

            df_out = (self.cum_flow().merge(
                right = self.prep_credit(), how = "inner", on = ["date", "security"]).
                melt(id_vars = ["date", "security"]).
                groupby(["security", "variable"]).
                apply(self._get_trend, self.trend_window).
                reset_index(drop = True).
                assign(signal = lambda x: np.where(x.variable == "flow", x.signal, -x.signal)))
            
            if verbose == True: print("Saving Data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_trend_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                z_win      = window,
                roll_mean  = lambda x: x.signal.ewm(span = window, adjust = False).mean(),
                roll_std   = lambda x: x.signal.ewm(span = window, adjust = False).std(),
                z_score    = lambda x: (x.signal - x.roll_mean) / x.roll_std,
                lag_zscore = lambda x: x.z_score.shift()).
            drop(columns = ["roll_mean", "roll_std"]))
        
        return df_out
    
    def get_trend_zscore(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.signal_path, "CreditTrendZScore.parquet")
        
        try:
            
            if verbose == True: print("Trying to find trend z-score data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")
            df_out = (self.get_trend().rename(
                columns = {"window": "trend_win"}).
                groupby(["security", "variable"]).
                apply(self._get_trend_zscore, self.zscore_window).
                reset_index(drop = True))
            
            if verbose == True: print("Saving Data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

    def get_lag_zscore_signal(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "CreditLagZScoreSignal.parquet")
        
        try:
            
            if verbose == True: print("Trying to find lagged trend z-score signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")
        
            df_out = (self.get_trend_zscore().pivot(
                index = ["date", "security"], columns = "variable", values = "lag_zscore").
                reset_index().
                melt(id_vars = ["date", "security", "flow"]).
                dropna().
                assign(spread = lambda x: x.flow - x.value))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out        

def main() -> None:
    
    Signal().get_trend(verbose = True)
    Signal().get_trend_zscore(verbose = True)
    Signal().get_lag_zscore_signal(verbose = True)
    
if __name__ == "__main__": main()