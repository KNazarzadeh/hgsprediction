import pandas as pd
import numpy as np
from ptpython.repl import embed

###############################################################################
###############################################################################
# This class define all required features from data:
def define_features(feature_type):
    
    features = []
    
    if feature_type == "anthropometrics":
        features = define_anthropometrics_features()
            
    elif feature_type == "anthropometrics_gender":
        features = define_anthropometrics_features().append(define_gender_features())

    elif feature_type == "anthropometrics_age":
        features = define_anthropometrics_features() + define_age_features()

    # elif feature_type == "behavioral":
    #     features = define_behavioral_features()
        
    # elif feature_type == "behavioral_gender":
    #     features = define_behavioral_features() + define_gender_features()
        
    # elif feature_type == "anthropometrics_behavioral":
    #     features = define_anthropometric_features() + define_behavioral_features()   
                    
    # elif feature_type == "anthropometrics_behavioral_gender":
    #     features = define_anthropometric_features() + define_behavioral_features() + extract_gender_features()
            
    return features
###############################################################################
def define_anthropometrics_features():
    
    anthropometric_features = [
        # ====================== Body size measures ======================
        "bmi",     # '21001',  # Body mass index (BMI)
        "height",  # '50',  # Standing height
        "waist_to_hip_ratio",  # Waist('48') to Hip('49') circumference Ratio
    ]
    
    return anthropometric_features
###############################################################################    
# define anthropometric and gender features from the data.
def define_gender_features():
    """define Gender Features.

    Parameters
    ----------
    None

    Returns
    --------
    gender_features : list of lists
        List of gender features.
    """
    gender_features = [
        # ============================ Gender ============================
        "gender",  # '31',
        ]
    return gender_features
###############################################################################    
# define anthropometric and age features from the data.
def define_age_features():
    """define Age Features.
    and add "Age" column to dataframe

    Parameters
    ----------

    Return
    ----------

    """
    age_features = [
        # ====================== Assessment attendance ======================
    "age",     # 'Age',  # Age at first Visit the assessment centre
                # '21003',  # Age when attended assessment centre
    ]

    return age_features

###############################################################################