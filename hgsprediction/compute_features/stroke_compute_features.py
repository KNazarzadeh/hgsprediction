
import pandas as pd
import numpy as np
from ptpython.repl import embed

###############################################################################
class Features:
    def __init__(self, df: pd.DataFrame, mri_status, stroke_cohort, visit_session):
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
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(mri_status, str), "mri_status must be a string!"
        assert isinstance(stroke_cohort, str), "stroke_cohort must be a string!"
        assert isinstance(visit_session, int), "visit_session must be a integer!"
        
        if visit_session == 1:
            self.session_column = f"1st_{stroke_cohort}-stroke_session"
        elif visit_session == 2:
            self.session_column = f"2nd_{stroke_cohort}-stroke_session"
        elif visit_session == 3:
            session_column = df[f"3rd_{stroke_cohort}-stroke_session"]
        elif visit_session == 4:
            session_column = df[f"4th_{stroke_cohort}-stroke_session"]

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
