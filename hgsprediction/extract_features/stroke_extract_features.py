import pandas as pd
import numpy as np
from ptpython.repl import embed

###############################################################################
class StrokeExtractFeatures:
    def __init__(self, 
                 df, 
                 mri_status,
                 stroke_cohort, 
                 visit_session,
                 feature_type):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df
        self.mri_status = mri_status
        self.stroke_cohort = stroke_cohort
        self.visit_session = visit_session
        self.feature_type = feature_type
        
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
    # This class extract all required features from data:
    def extract_features(self, df):
        
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        feature_type = self.feature_type
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        features = []
        if feature_type == "anthropometrics":
            features = self.extract_anthropometrics_features()
                
        elif feature_type == "anthropometrics_gender":
            features = self.extract_anthropometrics_features().append(self.extract_gender_features())

        elif feature_type == "anthropometrics_age":
            features = self.extract_anthropometrics_features() + self.extract_age_features()

        # elif feature_type == "behavioral":
        #     features = self.extract_behavioral_features()
            
        # elif feature_type == "behavioral_gender":
        #     features = self.extract_behavioral_features() + self.extract_gender_features()
            
        # elif feature_type == "anthropometrics_behavioral":
        #     features = self.extract_anthropometric_features() + self.extract_behavioral_features()   
                        
        # elif feature_type == "anthropometrics_behavioral_gender":
        #     features = self.extract_anthropometric_features() + self.extract_behavioral_features() + self.extract_gender_features()
        
        df = df[features]
        
        return df, features
###############################################################################
    def extract_anthropometrics_features(self):
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        bmi = session_column.replace(substring_to_remove, "bmi")
        height = session_column.replace(substring_to_remove, "height")
        whr = session_column.replace(substring_to_remove, "waist_to_hip_ratio")

        anthropometric_features = [
            # ====================== Body size measures ======================
            bmi,     # '21001',  # Body mass index (BMI)
            height,  # '50',  # Standing height
            whr,     #'waist_to_hip_ratio',  # Waist to Hip circumference Ratio
        ]
        
        return anthropometric_features
###############################################################################    
    # Extract anthropometric and gender features from the data.
    def extract_gender_features(self):
        """Extract Gender Features.

        Parameters
        ----------
        None

        Returns
        --------
        gender_features : list of lists
            List of gender features.
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(session_column, str), "session_column must be a string!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        gender = session_column.replace(substring_to_remove, "gender")

        gender_features = [
            # ============================ Gender ============================
            gender,  # '31',
            ]
        return gender_features
###############################################################################    
    # Extract anthropometric and age features from the data.
    def extract_age_features(self):
        """Extract Age Features.
        and add "Age" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: Age
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(session_column, str), "session_column must be a string!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        age = session_column.replace(substring_to_remove, "age")
        age_features = [
            # ====================== Assessment attendance ======================
        age,     # 'Age',  # Age at first Visit the assessment centre
                    # '21003',  # Age when attended assessment centre
        ]

        return age_features

###############################################################################