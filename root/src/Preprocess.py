# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:52:17 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   DataCollect import DataCollect

class Preprocess(DataCollect):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.preprocess_path = os.path.join(self.data_path, "preprocess")
        if os.path.exists(self.preprocess_path) == False: os.makedirs(self.preprocess_path)
        
        self.window = 10
    
    def prep_credit(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.preprocess_path, "LogCreditFundamentals.parquet")
        
        try:
            
            if verbose == True: print("Trying to find log credit data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")
        
            df_out = (self.combine_index_oas().pivot(
                index = ["date", "security"], columns = "variable", values = "replace_val").
                apply(lambda x: np.log(x)).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def cum_flow(self, verbose = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.preprocess_path, "CumulativeFlow.parquet")
        
        try:
            
            if verbose == True: print("Trying to find log credit data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting data now")
        
            df_out = (self.get_flow_data().pivot(
                index = "date", columns = "path", values = "value").
                fillna(0).
                cumsum().
                reset_index().
                melt(id_vars = "date").
                rename(columns = {
                    "path" : "security",
                    "value": "flow"}))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out

def main() -> None:
           
    Preprocess().prep_credit(verbose = True)
    Preprocess().cum_flow(verbose = True)
    
if __name__ == "__main__": main()