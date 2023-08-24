import pandas as pd
import numpy as np


def extract_data(stroke_cohort, visit_session, features, target):
    
    if visit_session == "1":
        prefix = f"1st_{stroke_cohort}_"
    elif visit_session == "2":
        prefix = f"2nd_{stroke_cohort}_"
    elif visit_session == "3":
        prefix = f"3rd_{stroke_cohort}_"
    elif visit_session == "4":
        prefix = f"4th_{stroke_cohort}_"

    features_list = []
    target_list = []

    for item in features:
        features_list.append(prefix + item)
        
    for item in target:
        target_list.append(prefix + item)    
        
    df = pd.concat([df[features_list], df[target_list]], axis=1)

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
        
    df.columns = df.columns.str.replace(prefix, '')
    
    return df