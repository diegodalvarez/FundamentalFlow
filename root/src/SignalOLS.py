# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:08:18 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   tqdm import tqdm
from   Signal import Signal

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class SignalOLS(Signal):
    
    def __init__(self):
        
        super().__init__()
        self.ols_signal = os.path.join(self.data_path, "SignalOLS")
        if os.path.exists(self.ols_signal) == False: os.makedirs(self.ols_signal)
        
        self.sample_size = 0.3
        self.num_sims    = 10_000
    
    def _get_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(px_rtn  = lambda x: x.px.pct_change()))
        
        return df_out
        
    def _get_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
    
        model = (sm.OLS(
            endog = df.px_rtn,
            exog  = sm.add_constant(df.spread)).
            fit())
        
        df_val = (model.params.to_frame(
            name = "param_val").
            reset_index())
        
        df_tmp = (model.pvalues.to_frame(
            name = "pvalue").
            reset_index().
            merge(right = df_val, how = "inner", on = ["index"]))
        
        df_out = (df.assign(
            index = "spread").
            merge(right = df_tmp, how = "inner", on = ["index"]))
 
        return df_out
     
    def full_sample_ols(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.ols_signal, "FullSampleSignal.parquet")
        
        try:
            
            if verbose == True: print("Trying to find Full Sample Signal signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")
        
            df_rtn = (Signal().get_bond_px().groupby(
                "security").
                apply(self._get_rtn).
                reset_index(drop = True).
                drop(columns = ["px"]).
                assign(date = lambda x: pd.to_datetime(x.date)))
            
            df_out = (self.get_lag_zscore_signal().merge(
                right = df_rtn, how = "inner", on = ["date", "security"]).
                assign(spread = lambda x: np.where(
                    (x.spread == np.inf) | (x.spread == -np.inf), 
                    np.nan, 
                    x.spread)).
                dropna().
                groupby(["security", "variable"]).
                apply(self._get_ols).
                reset_index(drop = True).
                drop(columns = ["index"]).
                assign(signal_rtn = lambda x: np.sign(x.param_val * x.spread) * x.px_rtn))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _beta(self, df: pd.DataFrame, sample_size: float) -> pd.DataFrame: 
    
        df_tmp = df.sample(frac = sample_size)
        df_out = (sm.OLS(
            endog = df_tmp.px_rtn,
            exog  = sm.add_constant(df_tmp.spread)).
            fit().
            params.
            to_frame(name = "param_val").
            reset_index().
            query("index == 'spread'"))
        
        return df_out
        
    def _run_beta_sim(self, df: pd.DataFrame, sample_size: float, num_sims: int) -> pd.DataFrame: 
        
        df_out = (pd.concat(
            [self._beta(df, sample_size).assign(sim = i + 1) 
            for i in tqdm(range(num_sims), desc = "Working on {} {}".format(df.name[0], df.name[1]))]))
        
        return df_out
    
    def bootstrapped_sample_ols(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ols_signal, "BootstrappedSampleParams.parquet")
        
        try:
            
            if verbose == True: print("Trying to find Full Sample Signal signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")

            df_out = (self.full_sample_ols().drop(
                columns = ["pvalue", "param_val"]).
                groupby(["security", "variable"]).
                apply(self._run_beta_sim, self.sample_size, self.num_sims).
                reset_index().
                drop(columns = ["level_2"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _expanding_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        model = (RollingOLS(
            endog     = df_tmp.px_rtn,
            exog      = sm.add_constant(df_tmp.spread),
            expanding = True).
            fit())
        
        df_params = (model.params[
            ["spread"]].
            rename(columns = {"spread": "beta"}))
        
        df_pvalue = (pd.DataFrame(
            data    = model.pvalues,
            columns = ["const", "spread"],
            index   = df_tmp.index)
            [["spread"]].
            rename(columns = {"spread": "pvalue"}))
        
        df_out = (df_params.merge(
            right = df_pvalue, how = "inner", on = ["date"]).
            dropna().
            merge(right = df_tmp, how = "inner", on = ["date"]).
            assign(lag_beta = lambda x: x.beta.shift()))
        
        return df_out
    
    def expanding_ols(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ols_signal, "ExpandingSampleSignal.parquet")
        
        try:
            
            if verbose == True: print("Trying to find Full Sample Signal signal data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")
        
            df_rtn = (Signal().get_bond_px().groupby(
                "security").
                apply(self._get_rtn).
                reset_index(drop = True).
                drop(columns = ["px"]).
                assign(date = lambda x: pd.to_datetime(x.date)))
            
            df_out = (self.get_lag_zscore_signal().merge(
                right = df_rtn, how = "inner", on = ["date", "security"]).
                groupby(["security", "variable"]).
                apply(self._expanding_ols).
                drop(columns = ["variable", "security"]).
                reset_index().
                assign(signal_rtn = lambda x: np.sign(x.lag_beta * x.spread) * x.px_rtn))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")

        return df_out

        
def main() -> None:
    
    SignalOLS().full_sample_ols(verbose = True)
    SignalOLS().bootstrapped_sample_ols(verbose = True)
    SignalOLS().expanding_ols(verbose = True)
    
if __name__ == "__main__": main()

