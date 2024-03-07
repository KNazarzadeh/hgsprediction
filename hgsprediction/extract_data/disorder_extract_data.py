import pandas as pd
import numpy as np
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def extract_data(df, population, features, extend_features, target, disorder_subgroup, visit_session):

    # Convert target and features for specific session
    target = [col for col in df.columns if col.endswith(target)]

    features= [col for col in df.columns for item in features if item in col]

    # Append target_list to feature_list
    features.extend(target)
    
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}_"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}_"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}_"
    
        
    features_extra_list = ["days", 
                            "years",
                            "handedness", 
                            "hgs_dominant", 
                            "hgs_nondominant",
                            "hgs_dominant_side",
                            "hgs_nondominant_side",
                            "session",
                            ]
    
    extend_features = [f"{item}-" for item in extend_features]
    extend_features = extend_features + ["1707-0.0", 
                                            "1707-1.0",
                                            "1707-2.0",
                                            "gender",
                                            "handness",
                                            "46-", 
                                            "47-",
                                            "followup",
                                            ]
    if population == "stroke":
        extend_features = extend_features + ["stroke_subtype", "42006-0.0"]
    elif population == "parkinson":
        extend_features = extend_features + ["131022-0.0"]
    if population == "depression":
        extend_features = extend_features + ["130894-0.0"]
        
    features_extra_list = [col for col in df.columns if any(col.startswith(prefix) and col.endswith(item) for item in features_extra_list)]
    
    extend_features_list = [col for item in extend_features for col in df.columns if col.startswith(item)]
    
    all_featurtes = features + features_extra_list + extend_features_list
    # Extract the specified feature and target columns
    # Drop rows with NaN values in the combined feature and target columns
    df_tmp = df.loc[:, all_featurtes].dropna(subset=features) 
    
    if disorder_subgroup == f"pre-{population}":
        df_extracted = df_tmp[[col for col in df_tmp.columns if f"post-{population}" not in col]]

    elif disorder_subgroup == f"post-{population}":
        df_extracted = df_tmp[[col for col in df_tmp.columns if f"pre-{population}" not in col]]
    
    # Filter columns that start with the specified prefix
    filtered_columns = [col for col in df_extracted.columns if col in features]

    # Remove the prefix from selected column names
    for col in filtered_columns:
        new_col_name = col.replace(prefix, "")
        df_extracted.rename(columns={col: new_col_name}, inplace=True)

    return df_extracted