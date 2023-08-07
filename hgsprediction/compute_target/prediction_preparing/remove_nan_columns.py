
import numpy as np
import pandas as pd

# Remove columns that all values are NaN
def remove_nan_columns(df):
    nan_cols = df.columns[df.isna().all()].tolist()
    df = df.drop(nan_cols, axis=1)
    
    return df
