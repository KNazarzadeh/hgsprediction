
import pandas as pd
from .features_extractor import FeaturesExtractor

def run_features_extraction(feature_type, mri_status):
    
        extractor = FeaturesExtractor(feature_type,mri_status)

        if feature_type == "anthropometrics":
            return extractor.extract_anthropometric_features()
            
        elif feature_type == "anthropometrics_gender":
            return extractor.extract_anthropometric_features().append(extractor.extract_gender_features())

        elif feature_type == "anthropometrics_age":
            return extractor.extract_anthropometric_features().append(extractor.extract_age_features())
            
        elif feature_type == "behavioral":
            return extractor.extract_behavioral_features()
            
        elif feature_type == "behavioral_gender":
            return extractor.extract_behavioral_features().append(extractor.extract_gender_features())
            
        elif feature_type == "anthropometrics_behavioral":
            return extractor.extract_anthropometric_features().append(extractor.extract_behavioral_features())   
                        
        elif feature_type == "anthropometrics_behavioral_gender":
            return extractor.extract_anthropometric_features() + extractor.extract_behavioral_features() + extractor.extract_gender_features()

def define_features(feature_type, mri_status):
    features = run_features_extraction(feature_type, mri_status)
    if mri_status == "nonmri":
        return features.rename(columns={"Age1stVisit": "Age"}, inplace=True)
    elif mri_status == "mri":
        return features.rename(columns={"AgeAtScan": "Age"}, inplace=True)