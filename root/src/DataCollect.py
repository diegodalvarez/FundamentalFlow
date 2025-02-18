# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:33:55 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as web

class DataCollect: 
    
    def __init__(self) -> None:
        
        self.dir       = os.path.dirname(os.path.abspath(__file__))  
        self.root_path = os.path.abspath(
            os.path.join(os.path.abspath(
                os.path.join(self.dir, os.pardir)), os.pardir))
        
        self.data_path = os.path.join(self.root_path, "data")
        self.raw_path  = os.path.join(self.data_path, "raw")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path) == False: os.makedirs(self.raw_path)
        
        self.etf_flow_path  = r"C:\Users\Diego\Desktop\app_prod\research\ETFFlow\data\prep"
        self.bond_fund_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\ETFIndices\BondPricing" 
        self.credit_indices = r"C:\Users\Diego\Desktop\app_prod\BBGData\credit_indices_data"
        
        self.renamer = {
            "YAS_YLD_SPREAD"         : "yas_sprd",
            "YAS_BOND_YLD"           : "yas_yls",
            "YAS_ISPREAD_TO_GOVT"    : "yas_ispread"}
        
        self.bad_attributes = ["YAS_MOD_DUR", "AVERAGE_WEIGHTED_COUPON"]
        
        self.bad_tickers = [
            "GOVT", "IEI", "SHY", "SHV", "IEF", "TLT", "TIP", "SGOV", "IEAC"]
        
        self.clean_window = 30
        
    def get_flow_data(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.raw_path, "FlowData.parquet")
        
        try:
            
            if verbose == True: print("Trying to find flow data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            read_in_path = os.path.join(self.etf_flow_path, "FlowData.parquet")
            
            if verbose == True: print("Couldn't find data, collecting it now")
            df_out = pd.read_parquet(path = read_in_path, engine = "pyarrow")
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_bond_fundamental(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "RawBondFundamentals.parquet")
        
        try:
            
            if verbose == True: print("Trying to find flow data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find bond fundamentals, collecting data now")
            
            paths = ([
                os.path.join(self.bond_fund_path, path)
                for path in os.listdir(self.bond_fund_path)])
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                query("variable != @self.bad_attributes").
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                pivot(index = ["date", "security"], columns = "variable", values = "value").
                rename(columns = self.renamer).
                reset_index().
                query("security != @self.bad_tickers"))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_bond_px(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "BondPX.parquet")
        
        try:
            
            if verbose == True: print("Trying to find flow data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find etf prices, collecting data now")
            tickers = (self.get_bond_fundamental().security.drop_duplicates().sort_values().to_list())
            df_out  = (yf.download(
                tickers = tickers)
                ["Adj Close"].
                reset_index().
                melt(id_vars = "Date").
                query("Ticker != @self.bad_tickers").
                dropna().
                rename(columns = {
                    "Date"  : "date",
                    "Ticker": "security",
                    "value" : "px"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _clean(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        
        df_out = (df.assign(
            diff_val = lambda x: x.value.diff(),
            mean_val    = lambda x: x.diff_val.mean(),
            std_val     = lambda x: x.diff_val.std(),
            z_score     = lambda x: np.abs((x.diff_val - x.mean_val) / x.std_val),
            tmp         = lambda x: np.where(x.z_score > 4, np.nan, x.value),
            replace_val = lambda x: x.tmp.ewm(span = 2, adjust = False).mean()).
            drop(columns = ["diff_val", "mean_val", "std_val", "z_score", "tmp"]).
            rename(columns = {"replace_val": "new_val"}).
            assign(
                mean_val    = lambda x: x.new_val.rolling(window = window).mean(),
                std_val     = lambda x: x.new_val.rolling(window = window).std(),
                z_score     = lambda x: np.abs((x.new_val - x.mean_val) / x.std_val),
                tmp         = lambda x: np.where(x.z_score > 4, np.nan, x.new_val),
                replace_val = lambda x: x.tmp.ewm(span = 2, adjust = False).mean()).
            drop(columns = ["new_val", "mean_val", "std_val", "z_score", "tmp"]))
        
        return df_out
    
    def clean_bond_fundamental(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "CleanedBondFundamentals.parquet")
        
        try:
            
            if verbose == True: print("Trying to find flow data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find etf prices, collecting data now")
        
            df_out = (self.get_bond_fundamental().melt(
                id_vars = ["date", "security"]).
                groupby(["security", "variable"]).
                apply(self._clean, self.clean_window).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def combine_index_oas(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.raw_path, "CombinedBondFundamentals.parquet")
        
        try:
            
            if verbose == True: print("Trying to find flow data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find etf index OAS, collecting data now")
        
            read_in    = os.path.join(self.dir, "ticker_info.xlsx")
            df_tickers =(pd.read_excel(
                io = read_in, sheet_name = "bond_indices"))
            
            df_bbg = df_tickers.query("source == 'bbg'")
            
            bbg_paths = [
                os.path.join(self.credit_indices, ticker + ".parquet")
                for ticker in df_bbg.oas_ticker.drop_duplicates().to_list()]
            
            df_oas = (pd.read_parquet(
                path = bbg_paths, engine = "pyarrow").
                assign(
                    security   = lambda x: x.security.str.split(" ").str[0],
                    ending     = lambda x: x.security.str[-3:],
                    oas_ticker = lambda x: x.security.str[:-3]).
                query("ending == 'OAS'").
                drop(columns = ["security"]).
                merge(right = df_tickers, how = "inner", on = ["oas_ticker"]).
                drop(columns = ["ending", "oas_ticker", "variable", "source"]).
                assign(
                    variable    = "OAS",
                    replace_val = lambda x: x.value))
            
            start_date, end_date = df_oas.date.min().date(), df_oas.date.max().date()
            
            df_fred = df_tickers.query("source =='fred'")
            df_fred_oas = (web.DataReader(
                name        = df_fred.oas_ticker.to_list(), 
                data_source = "fred",
                start       = start_date,
                end         = end_date).
                reset_index().
                melt(id_vars = ["DATE"]).
                rename(columns = {
                    "DATE"    : "date",
                    "variable": "oas_ticker"}).
                merge(right = df_fred, how = "inner", on = ["oas_ticker"]).
                drop(columns = ["source", "oas_ticker"]).
                assign(
                    variable    = "OAS",
                    date        = lambda x: pd.to_datetime(x.date),
                    replace_val = lambda x: x.value))
            
            df_out = (pd.concat(
                [df_oas, self.clean_bond_fundamental(), df_fred_oas]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:

    DataCollect().get_flow_data(verbose = True)
    DataCollect().get_bond_fundamental(verbose = True)
    DataCollect().get_bond_px(verbose = True)
    DataCollect().clean_bond_fundamental(verbose = True)
    DataCollect().combine_index_oas(verbose = True)
    
if __name__ == "__main__": main()