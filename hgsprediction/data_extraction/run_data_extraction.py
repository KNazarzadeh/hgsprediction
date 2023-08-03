import pandas as pd
from hgsprediction.features_extraction import run_features_extraction

def data_extractor(data, feature_type, target, mri_status):
    
    features = run_features_extraction(feature_type, mri_status)
    if mri_status == "nonmri":
        features = add_suffix_to_features(features, "-0.0", exception_feature="Age1stVisit")
    elif mri_status == "mri":
        features = add_suffix_to_features(features, "-2.0", exception_feature="AgeAtScan")
    
    target = [col for col in data if col.startswith(f"hgs_({target})")][0]
    
    data = pd.concat([data[features], data[target]], axis=1)
    
    
    if mri_status == "nonmri":
        data = rename_column(data, "Age1stVisit", "Age")
        return remove_suffix_from_columns(data, column_suffix="-0.0")
    elif mri_status == "mri":
        data = rename_column(data, "AgeAtScan", "Age")        
        return remove_suffix_from_columns(data, column_suffix="-2.0")

def add_suffix_to_features(features_list, feature_suffix, exception_feature="Age1stVisit"):
    features = []
    for feature in features_list:
        if feature != exception_feature:
            features.append(f"{feature}{feature_suffix}")
    return features

def remove_suffix_from_columns(data, column_suffix):
    new_columns = [col[:-len(column_suffix)] if col.endswith(column_suffix) else col for col in data.columns]
    data.columns = new_columns
    return data

def rename_column(data, old_column_name, new_column_name):
    data.rename(columns={old_column_name: new_column_name}, inplace=True)
    return data