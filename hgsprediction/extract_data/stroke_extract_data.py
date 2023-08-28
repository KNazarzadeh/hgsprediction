import pandas as pd
import numpy as np


def extract_data(df, stroke_cohort, visit_session, features, target):
        
    features_columns = [col for col in df.columns if any(col.endswith(item) for item in features)]
    target_columns = [col for col in df.columns if col.endswith(target)]
    
    df = pd.concat([df[features_columns], df[target_columns]], axis=1)
    
    df = rename_column_names(df, stroke_cohort, visit_session)               

    df = df.dropna()
    
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