# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:59:28 2025

@author: Diego
"""

from SignalOLS import SignalOLS

def main() -> None:
    
    SignalOLS().get_flow_data(verbose = True)
    SignalOLS().get_bond_fundamental(verbose = True)
    SignalOLS().get_bond_px(verbose = True)
    SignalOLS().clean_bond_fundamental(verbose = True)
    SignalOLS().combine_index_oas(verbose = True)
    
    SignalOLS().prep_credit(verbose = True)
    SignalOLS().cum_flow(verbose = True)
    
    SignalOLS().get_trend(verbose = True)
    SignalOLS().get_trend_zscore(verbose = True)
    
    SignalOLS().full_sample_ols(verbose = True)
    #SignalOLS().bootstrapped_sample_ols(verbose = True)
    SignalOLS().expanding_ols(verbose = True)
    
#if __name__ == "__main__": main()