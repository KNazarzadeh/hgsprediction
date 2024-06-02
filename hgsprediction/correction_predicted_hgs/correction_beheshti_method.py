
import numpy as np
import pandas as pd

def beheshti_correction_method(
    df,
    target,
    true_hgs,
    predicted_hgs,
    slope, 
    intercept,
):
    #Beheshti Method:
    
    offset = slope * true_hgs + intercept
    
    df.loc[:, f"{target}_corrected_predicted"] = predicted_hgs + offset
    
    # Calculate Corrected Delta
    df.loc[:, f"{target}_corrected_delta(true-predicted)"] =  true_hgs - df.loc[:, f"{target}_corrected_predicted"]
    
    return df