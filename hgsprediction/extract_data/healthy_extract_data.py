import pandas as pd
import numpy as np
from ptpython.repl import embed


def extract_data(df, mri_status, features, target, session):
    
    features_list = features
    add_extra_features = ["age", "gender", "handedness", "dominant", "nondominant"]
    for item in add_extra_features:
        if item not in features:
           features_list = [item] + features_list

    features_columns = [col for col in df.columns if any(item in col for item in features_list)]
    target_columns = [col for col in df.columns if col.startswith(target)]
    
    df = pd.concat([df[features_columns], df[target_columns]], axis=1)
    
    df = df.dropna(subset=[col for col in df.columns if any(item in col for item in features)])

    df = rename_column_names(df, mri_status, session)               

    return df

def rename_column_names(df, mri_status, session):
    
    # if mri_status == "nonmri":
    #     prefix = "-0.0"
    # elif mri_status == "mri":
    #     prefix = "-2.0"
    if session == "0":
        prefix = "-0.0"
    elif session == "2":
        prefix = "-2.0"
    elif session == "3":
        prefix = "-3.0"
    df.columns = df.columns.str.replace(prefix, '')
    
    return df