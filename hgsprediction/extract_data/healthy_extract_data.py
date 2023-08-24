import pandas as pd
import numpy as np


def extract_data(mri_status, features, target):
    
    if mri_status == "nonmri":
        prefix = f"_0.0"
    elif mri_status == "mri":
        prefix = f"_2.0"

    features_list = []
    target_list = []

    for item in features:
        features_list.append(item + prefix)
        
    for item in target:
        target_list.append(item + prefix)    
        
    df = pd.concat([df[features_list], df[target_list]], axis=1)

    df = rename_column_names(df, mri_status)               

    return df

def rename_column_names(df, mri_status):
    
    if mri_status == "nonmri":
        prefix = f"_0.0"
    elif mri_status == "mri":
        prefix = f"_2.0"

    df.columns = df.columns.str.replace(prefix, '')
    
    return df