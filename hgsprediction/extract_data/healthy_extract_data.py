import pandas as pd
import numpy as np


def extract_data(df, mri_status, features, target):
    
    features_columns = [col for col in df.columns if any(col.startswith(item) for item in features)]
    target_columns = [col for col in df.columns if col.startswith(target)]
    
    df = pd.concat([df[features_columns], df[target_columns]], axis=1)
    
    df = rename_column_names(df, mri_status)               

    df = df.dropna()
    
    return df

def rename_column_names(df, mri_status):
    
    if mri_status == "nonmri":
        prefix = "-0.0"
    elif mri_status == "mri":
        prefix = "-2.0"

    df.columns = df.columns.str.replace(prefix, '')
    
    return df