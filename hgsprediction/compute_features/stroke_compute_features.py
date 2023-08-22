
import pandas as pd
import numpy as np
from ptpython.repl import embed

###############################################################################
class StrokeFeaturesComputing:
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
    def calculate_bmi(self, df):
        """Calculate coressponding BMI
        and add "BMI" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: BMI
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Add a new column 'new_column'
        bmi = session_column.replace(substring_to_remove, "bmi")
        
        df[bmi] = df.apply(lambda row: row[f"21001-{row[session_column]}"], axis=1)

        return df

###############################################################################
    def calculate_height(self, df):
        """Calculate coressponding Height
        and add "Height" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: Height
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        height = session_column.replace(substring_to_remove, "height")
        
        df[height] = df.apply(lambda row: row[f"50-{row[session_column]}"], axis=1)

        return df
    
    
###############################################################################
    def calculate_waist_to_hip_ratio(self, df):
        """Calculate coressponding waist_to_hip_ratio
        and add "waist_to_hip_ratio" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: waist_to_hip_ratio
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        whr = session_column.replace(substring_to_remove, "waist_to_hip_ratio")

        df[whr] = df.apply(lambda row: row[f"48-{row[session_column]}"]/row[f"49-{row[session_column]}"], axis=1)
        
        return df

###############################################################################
    def calculate_age(self, df):
        """Calculate coressponding Age
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
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        age = session_column.replace(substring_to_remove, "age")
        for idx in df.index:
            session = df.loc[idx, session_column]
            if session == 0.0:
                df.loc[idx, age] = df.loc[idx, "Age1stVisit"]
            elif session == 1.0:
                df.loc[idx, age] = df.loc[idx, "AgeRepVisit"]
            elif session == 2.0:
                df.loc[idx, age] = df.loc[idx, "AgeAtScan"]
            elif session == 3.0:
                df.loc[idx, age] = df.loc[idx, "AgeAt2ndScan"]
        
        return df    

###############################################################################
    def calculate_days(self, df):
        """Calculate coressponding Height
        and add "Height" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: Height
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        days = session_column.replace(substring_to_remove, "days")
       
        df[days] = df.apply(lambda row: row[f"followup_days-{row[session_column]}"], axis=1)

        return df

###############################################################################    
    def calculate_neuroticism_score(self, df):
        """Calculate neuroticism score
        and add "neuroticism_score" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: neuroticism score
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        # ------- Neuroticism Fields and Data-Codings -------
        # All Fields useing same Data-Coding (100349)
        # Data-Coding: 100349
        #        1	Yes
        #        0	No
        #       -1	Do not know
        #       -3	Prefer not to answer
        # ------------------------------------
        # ------- Neuroticism Fields -------
        # Mood swings
        # '1920',   Data-Coding: 100349
        # Miserableness
        # '1930',   Data-Coding: 100349
        # Irritability
        # '1940',   Data-Coding: 100349
        # Sensitivity / hurt feelings
        # '1950',   Data-Coding: 100349
        # Fed-up feelings
        # '1960',   Data-Coding: 100349
        # Nervous feelings
        # '1970',   Data-Coding: 100349
        # Worrier / anxious feelings
        # '1980',   Data-Coding: 100349
        # Tense / 'highly strung'
        # '1990',   Data-Coding: 100349
        # Worry too long after embarrassment
        # '2000',   Data-Coding: 100349
        # Suffer from 'nerves'
        # '2010',   Data-Coding: 100349
        # Loneliness, isolation
        # '2020',   Data-Coding: 100349
        # Guilty feelings
        # '2030',   Data-Coding: 100349
        # -----------------------------------------------------------
        neuroticism_fields = [
                    '1920',     # Data-Coding: 100349
                    '1930',     # Data-Coding: 100349
                    '1940',     # Data-Coding: 100349
                    '1950',     # Data-Coding: 100349
                    '1960',     # Data-Coding: 100349
                    '1970',     # Data-Coding: 100349
                    '1980',     # Data-Coding: 100349
                    '1990',     # Data-Coding: 100349
                    '2000',     # Data-Coding: 100349
                    '2010',     # Data-Coding: 100349
                    '2020',     # Data-Coding: 100349
                    '2030',     # Data-Coding: 100349
        ]
        
        for idx in df.index:
            session = df.loc[idx, session_column]
            # Add corresponding intences/session that we are looking for 
            # to the list of fields as suffix:
            neuroticism_fields_tmp = [item + f"-{session}" for item in neuroticism_fields]
            # ------------------------------------
            # Add new column "neuroticism_score" by the following process: 
            # Find core_fields answered greather than 0.0
            # And Calculate Sum with min_count parameter=9:
            # df.where is replacing all negative values with NaN
            # min_count=9, means calculate Sum if 
            # at leaset 9 of the fields are answered:
            df.loc[idx, "neuroticism_score"] = \
                df.loc[idx, neuroticism_fields_tmp].where(df.loc[idx, neuroticism_fields_tmp] >= 0.0).sum(axis=0, min_count=9)

        return df

###############################################################################
###############################################################################
############################## Remove NaN coulmns #############################
# Remove columns if their values are all NAN
###############################################################################
# Remove columns that all values are NaN
    def remove_nan_columns(self, df):
        """Remove columns with all NAN values
      
        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
        """ 

        nan_cols = df.columns[df.isna().all()].tolist()
        df = df.drop(nan_cols, axis=1)
        
        return df
    
###############################################################################