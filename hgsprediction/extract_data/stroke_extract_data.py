import pandas as pd
import numpy as np
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def extract_data(df, stroke_cohort, visit_session, features, target):
    
    features_list = features
    add_extra_features = ["gender", "age", "days", "years", "handedness"]
    for item in add_extra_features:
        if item not in features:
           features_list = [item] + features_list

    features_columns = [col for col in df.columns if any(col.endswith(item) for item in features_list)]
    
    target_columns = [col for col in df.columns if col.endswith(target)]

    df = pd.concat([df[features_columns], df[target_columns]], axis=1)
    
    df = df.dropna(subset=[col for col in df.columns if any(item in col for item in features)])
    
    df = rename_column_names(df, stroke_cohort, visit_session) 
    
    return df

def rename_column_names(df, stroke_cohort, visit_session):
    
    if visit_session == "1":
        prefix = f"1st_{stroke_cohort}_"
    elif visit_session == "2":
        prefix = f"2nd_{stroke_cohort}_"
    elif visit_session == "3":
        prefix = f"3rd_{stroke_cohort}_"
    elif visit_session == "4":
        prefix = f"4th_{stroke_cohort}_"
    
    df.columns = df.columns.str.replace(prefix, "")
    
    return df