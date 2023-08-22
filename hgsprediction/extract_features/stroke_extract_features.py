import pandas as pd
import numpy as np
from ptpython.repl import embed

###############################################################################
class StrokeExtractFeatures:
    def __init__(self, 
                 df, 
                 mri_status,
                 feature_type,
                 stroke_cohort, 
                 visit_session):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df
        self.mri_status = mri_status
        self.feature_type = feature_type
        self.stroke_cohort = stroke_cohort
        self.visit_session = visit_session
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(mri_status, str), "mri_status must be a string!"
        assert isinstance(stroke_cohort, str), "stroke_cohort must be a string!"
        assert isinstance(visit_session, str), "visit_session must be a integer!"

        if visit_session == "1":
            self.session_column = f"1st_{stroke_cohort}_session"
        elif visit_session == "2":
            self.session_column = f"2nd_{stroke_cohort}_session"
        elif visit_session == "3":
            self.session_column = f"3rd_{stroke_cohort}_session"
        elif visit_session == "4":
            self.session_column = f"4th_{stroke_cohort}_session"
    
###############################################################################
    def extract_anthropometrics_features(self, df):
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        bmi = session_column.replace(substring_to_remove, "bmi")
        height = session_column.replace(substring_to_remove, "height")
        whr = session_column.replace(substring_to_remove, "waist_to_hip_ratio")

        df = df[[bmi, height, whr]]
        
        return df
###############################################################################
    def extract_anthropometrics_age_features(self, df):
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        bmi = session_column.replace(substring_to_remove, "bmi")
        height = session_column.replace(substring_to_remove, "height")
        whr = session_column.replace(substring_to_remove, "waist_to_hip_ratio")
        age = session_column.replace(substring_to_remove, "age")

        df = df[[bmi, height, whr, age]]
        
        return df
###############################################################################
    def extract_anthropometrics_gender_features(self, df):
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        bmi = session_column.replace(substring_to_remove, "bmi")
        height = session_column.replace(substring_to_remove, "height")
        whr = session_column.replace(substring_to_remove, "waist_to_hip_ratio")
        gender = "31-0.0"

        df = df[[bmi, height, whr, gender]]

        return df
        