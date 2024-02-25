import pandas as pd
import numpy as np
from ptpython.repl import embed


def extract_data(df, features, extend_features, target, mri_status, session):
    
    # Convert target and features for specific session
    target = f"{target}-{session}.0"

    feature_list = [f"{item}-{session}.0" for item in features]
    extend_features_list = [f"{item}-{session}.0" for item in extend_features]
    
    if mri_status == "nonmri":
        extra_columns_list = ["gender",
                            f"age_range-{session}.0",
                            f"1707-0.0", 
                            f"1707-1.0", 
                            f"handness",                             
                            f"handedness-{session}.0", 
                            f"46-{session}.0", 
                            f"47-{session}.0",
                            f"hgs_dominant-{session}.0", 
                            f"hgs_nondominant-{session}.0",
                            f"hgs_dominant_side-{session}.0",
                            f"hgs_nondominant_side-{session}.0",
                            ]
        
    
    elif mri_status == "mri":
        extra_columns_list = ["gender",
                            f"age_range-{session}.0",
                            f"1707-0.0", 
                            f"1707-1.0", 
                            f"1707-2.0", 
                            f"handness",                             
                            f"handedness-{session}.0", 
                            f"46-{session}.0", 
                            f"47-{session}.0",
                            f"hgs_dominant-{session}.0", 
                            f"hgs_nondominant-{session}.0",
                            f"hgs_dominant_side-{session}.0",
                            f"hgs_nondominant_side-{session}.0",
                            ]

    # Append target_list to feature_list
    feature_list.append(target)
    extra_columns_list.extend(extend_features_list)
    extra_columns_list.extend(feature_list)
    
    # Extract the specified feature and target columns
    # Drop rows with NaN values in the combined feature and target columns
    df_extracted = df.loc[:, extra_columns_list].dropna(subset=feature_list)
    
    # Remove "-{session}.0" from the end of feature_list column names
    # Substring to remove
    substring = f"-{session}.0"
    # Create a new list of column names with modifications
    new_columns = []
    for col in df_extracted.columns:
        if col in feature_list:
            new_columns.append(col.replace(substring, ''))
        else:
            new_columns.append(col)

    # Update the column names of the DataFrame
    df_extracted.columns = new_columns

    return df_extracted



    
    