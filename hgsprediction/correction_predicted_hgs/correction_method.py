
import numpy as np
import pandas as pd

def beheshti_correction_method(
    df,
    target,
    raw_hgs,
    predicted_hgs,
    slope, 
    intercept,
):
    #Beheshti Method:
    
    offset = slope * raw_hgs + intercept
    
    df.loc[:, f"{target}_corrected_predicted"] = predicted_hgs + offset
    
    # Calculate Corrected Delta
    df.loc[:, f"{target}_corrected_delta(true-predicted)"] =  raw_hgs - df.loc[:, f"{target}_corrected_predicted"]
    
    return df