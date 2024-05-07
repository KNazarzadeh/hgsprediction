import pandas as pd
import numpy as np
from ptpython.repl import embed


def extract_data(df, features, extend_features, feature_type, target, mri_status, session):
    
    # Convert target and features for specific session
    target_tmp = f"{target}-{session}.0"
    
    if mri_status == "nonmri":
        if feature_type == "behavioral":
            feature_list = [f"{item}-{session}.0" if item not in ['anxiety_score', 'depression_score', 'CIDI_score', 'neuroticism_score'] else item for item in features]
            extend_features_list =[]         
        else:
            feature_list = [f"{item}-{session}.0" for item in features]
            extend_features_list = [f"{item}-{session}.0" for item in extend_features]
        extra_columns_list = ["gender",
                            "1707-0.0", 
                            "1707-1.0", 
                            f"53-{session}.0",
                            "handness",                             
                            f"handedness-{session}.0", 
                            f"46-{session}.0", 
                            f"47-{session}.0",
                            f"hgs_dominant-{session}.0", 
                            f"hgs_nondominant-{session}.0",
                            f"hgs_dominant_side-{session}.0",
                            f"hgs_nondominant_side-{session}.0",
                            ]

    elif mri_status == "mri":
        if feature_type == "behavioral":
            features = [f"{item}-{session}.0" if item not in ['anxiety_score', 'depression_score', 'CIDI_score', 'neuroticism_score', 'general_happiness', 'health_happiness', 'belief_life_meaningful'] else item for item in features]
            feature_list = [f"{item}-0.0" if item in ['general_happiness', 'health_happiness', 'belief_life_meaningful'] else item for item in features]
            extend_features_list =[]         
        else:
            feature_list = [f"{item}-{session}.0" for item in features]
            extend_features_list = [f"{item}-{session}.0" for item in extend_features]
        extra_columns_list = ["gender",
                            "1707-0.0", 
                            "1707-1.0", 
                            "1707-2.0", 
                            f"53-{session}.0",
                            "handness",                             
                            f"handedness-{session}.0", 
                            f"46-{session}.0", 
                            f"47-{session}.0",
                            f"hgs_dominant-{session}.0", 
                            f"hgs_nondominant-{session}.0",
                            f"hgs_dominant_side-{session}.0",
                            f"hgs_nondominant_side-{session}.0",
                            ]
    # Append target_list to feature_list
    feature_list.append(target_tmp)
    extra_columns_list.extend(extend_features_list)
    extra_columns_list.extend(feature_list)
    
    # Extract the specified feature and target columns
    # Drop rows with NaN values in the combined feature and target columns
    df_extracted = df.loc[:, extra_columns_list].dropna(subset=feature_list)
    
    # Remove "-{session}.0" from the end of feature_list column names
    # Create a new list of column names with modifications
    new_columns = []
    for col in df_extracted.columns:
        if col in feature_list:
            new_columns.append(col.split('-')[0])
        else:
            new_columns.append(col)

    # Update the column names of the DataFrame
    df_extracted.columns = new_columns

    if target in ["hgs_dominant", "hgs_nondominant"]:
        df_extracted = df_extracted.loc[:, ~df_extracted.columns.duplicated()]
    
    return df_extracted



    
    