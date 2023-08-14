#!/usr/bin/env Disorderspredwp3

"""
Compute Target, Calculate and Add new columns based on corresponding Field-IDs and conditions

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

from ptpython.repl import embed

###############################################################################
class StrokeTargetComputing:
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
        self.stroke_cohort = stroke_cohort
        self.visit_session = visit_session
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(mri_status, str), "mri_status must be a string!"
        assert isinstance(feature_type, str), "feature_type must be a string!"        
        assert isinstance(stroke_cohort, str), "stroke_cohort must be a string!"
        assert isinstance(visit_session, int), "visit_session must be a integer!"
        
        if visit_session == 1:
            self.session_column = f"1st_{stroke_cohort}-stroke_session"
        elif visit_session == 2:
            self.session_column = f"2nd_{stroke_cohort}-stroke_session"
        elif visit_session == 3:
            self.session_column = f"3rd_{stroke_cohort}-stroke_session"
        elif visit_session == 4:
            self.session_column = f"4th_{stroke_cohort}-stroke_session"
###############################################################################
    def calculate_dominant_hgs(self, df):
        """Calculate dominant handgrip
        and add "hgs_dominant" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: Dominant hand Handgrip strength
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        for idx in df.index:
            session = df.loc[idx, session_column]
            if (session == 1.0) | (session == 3.0):
                session = 0.0

            # hgs_left field-ID: 46
            # hgs_right field-ID: 47
            # ------------------------------------
            # ------- Handedness Field-ID: 1707
            # Data-Coding: 100430
            #           1	Right-handed
            #           2	Left-handed
            #           3	Use both right and left hands equally
            #           -3	Prefer not to answer
            # ------------------------------------
            # If handedness is equal to 1
            # Right hand is Dominant
            # Find handedness equal to 1:
            if df.loc[idx,f"1707-{session}"] == 1.0:
                # Add and new column "hgs_dominant"
                # And assign Right hand HGS value:
                df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_dominant"] = \
                    df.loc[idx, f"47-{session}"]
            # ------------------------------------
            # If handedness is equal to 2
            # Right hand is Non-Dominant
            # Find handedness equal to 2:
            elif df.loc[idx,f"1707-{session}"] == 2.0:
                # Add and new column "hgs_dominant"
                # And assign Left hand HGS value:        
                df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_dominant"] = \
                    df.loc[idx, f"46-{session}"]
            # ------------------------------------
            # If handedness is equal to:
            # 3 (Use both right and left hands equally) OR
            # -3 (handiness is not available/Prefer not to answer) OR
            # NaN value
            # Dominant will be the Highest Handgrip score from both hands.
            # Find handedness equal to 3, -3 or NaN:
            if session == 0:
                if (df.loc[idx, f"1707-{session}"] == 3.0) | \
                    (df.loc[idx, f"1707-{session}"] == -3.0) | \
                    (df.loc[idx, f"1707-{session}"].isna()):
                    # Add and new column "hgs_dominant"
                    # And assign Highest HGS value among Right and Left HGS:        
                    df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_dominant"] = \
                        df.loc[idx, [f"46-{session}", f"47-{session}"]].max(axis=1)
            elif session == 2:
                if (df.loc[idx, f"1707-{session}"] == 3.0) | \
                    (df.loc[idx, f"1707-{session}"] == -3.0) | \
                    (df.loc[idx, f"1707-{session}"].isna()):
                    if df.loc[idx,"1707-0.0"] == 1.0:
                        # Add and new column "hgs_dominant"
                        # And assign Right hand HGS value:
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_dominant"] = \
                            df.loc[idx, "47-0.0"]
                    elif df.loc[idx,"1707-0.0"] == 2.0:
                        # Add and new column "hgs_dominant"
                        # And assign Left hand HGS value:
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_dominant"] = \
                            df.loc[idx, "46-0.0"]
                            
                    elif (df.loc[idx, "1707-0.0"] == 3.0) | \
                        (df.loc[idx, "1707-0.0"] == -3.0)| \
                        (df.loc[idx, "1707-0.0"].isna()):
                        # Add and new column "hgs_dominant"
                        # And assign Highest HGS value among Right and Left HGS:        
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_dominant"] = \
                            df.loc[idx, ["46-0.0", "47-0.0"]].max(axis=1) 

###############################################################################
    def calculate_nondominant_hgs(self, df):
        """Calculate dominant handgrip
        and add "hgs_nondominant" column to dataframe
        (Opposite Process of Dominant module)

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: NonDominant hand Handgrip strength
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        for idx in df.index:
            session = df.loc[idx, session_column]
            if (session == 1.0) | (session == 3.0):
                session = 0.0

            # hgs_left field-ID: 46
            # hgs_right field-ID: 47
            # ------------------------------------
            # ------- Handedness Field-ID: 1707
            # Data-Coding: 100430
            #           1	Right-handed
            #           2	Left-handed
            #           3	Use both right and left hands equally
            #           -3	Prefer not to answer
            # ------------------------------------
            # If handedness is equal to 1
            # Left hand is Non-Dominant
            # Find handedness equal to 1:
            if df.loc[idx,f"1707-{session}"] == 1.0:
                # Add and new column "hgs_nondominant"
                # And assign Left hand HGS value:
                df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_nondominant"] = \
                    df.loc[idx, f"46-{session}"]
            # ------------------------------------
            # If handedness is equal to 2
            # Right hand is Non-Dominant
            # Find handedness equal to 2:
            elif df.loc[idx,f"1707-{session}"] == 2.0:
                # Add and new column "hgs_nondominant"
                # And assign Right hand HGS value:        
                df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_nondominant"] = \
                    df.loc[idx, f"47-{session}"]
            # ------------------------------------
            # If handedness is equal to:
            # 3 (Use both right and left hands equally) OR
            # -3 (handiness is not available/Prefer not to answer) OR
            # NaN value
            # Non-Dominant will be the Lowest Handgrip score from both hands.
            # Find handedness equal to 3, -3 or NaN:
            if session == 0:
                if (df.loc[idx, f"1707-{session}"] == 3.0) | \
                    (df.loc[idx, f"1707-{session}"] == -3.0) | \
                    (df.loc[idx, f"1707-{session}"].isna()):
                    # Add and new column "hgs_nondominant"
                    # And assign Lowest HGS value among Right and Left HGS:        
                    df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_nondominant"] = \
                        df.loc[idx, [f"46-{session}", f"47-{session}"]].min(axis=1)
            elif session == 2:
                if (df.loc[idx, f"1707-{session}"] == 3.0) | \
                    (df.loc[idx, f"1707-{session}"] == -3.0) | \
                    (df.loc[idx, f"1707-{session}"].isna()):
                    if df.loc[idx,"1707-0.0"] == 1.0:
                        # Add and new column "hgs_nondominant"
                        # And assign Left hand HGS value:
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_nondominant"] = \
                            df.loc[idx, "46-0.0"]
                    elif df.loc[idx,"1707-0.0"] == 2.0:
                        # Add and new column "hgs_nondominant"
                        # And assign Right hand HGS value:
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_nondominant"] = \
                            df.loc[idx, "47-0.0"]
                            
                    elif (df.loc[idx, "1707-0.0"] == 3.0) | \
                        (df.loc[idx, "1707-0.0"] == -3.0)| \
                        (df.loc[idx, "1707-0.0"].isna()):
                        # Add and new column "hgs_nondominant"
                        # And assign Lowest HGS value among Right and Left HGS:        
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs_nondominant"] = \
                            df.loc[idx, ["46-0.0", "47-0.0"]].min(axis=1) 
    
        return df

###############################################################################
    def calculate_sum_hgs(self, df):
        """Calculate sum of Handgrips
        and add "hgs(L+R)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for: (HGS Left + HGS Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        df_left = self.calculate_left_hgs(df)
        df_right = self.calculate_left_hgs(df)

        # ------------------------------------
        # Add new column "hgs(L+R)" by the following process: 
        # hgs_left field-ID: 46
        # hgs_right field-ID: 47
        # sum of Handgrips (Left + Right)
        substring_to_remove = "session"
        df.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(L+R)"] = \
            df_left.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(left)"] + \
                df_right.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(right)"]

        return df

###############################################################################
    def calculate_left_hgs(self, df):
        """Calculate right and add "hgs(left)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for:
            HGS(Left)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        for idx in df.index:
            session = df.loc[idx, session_column]
            # hgs_left field-ID: 46
            # hgs_right field-ID: 47
            substring_to_remove = "session"
            df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(left)"] = \
                df.loc[idx, f"46-{session}"]
        return df

###############################################################################
    def calculate_right_hgs(self, df):
        """Calculate right and add "hgs(right)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for:
            HGS(Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        for idx in df.index:
            session = df.loc[idx, session_column]
            # hgs_left field-ID: 46
            # hgs_right field-ID: 47
            substring_to_remove = "session"
            df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(right)"] = \
                df.loc[idx, f"47-{session}"]
        return df

###############################################################################
    def calculate_sub_hgs(self, df):
        """Calculate subtraction of Handgrips
        and add "hgs(L-R)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for: (HGS Left - HGS Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        df_left = self.calculate_left_hgs(df)
        df_right = self.calculate_left_hgs(df)
        
        # Add new column "hgs(L-R)" by the following process: 
        # hgs_left field-ID: 46
        # hgs_right field-ID: 47
        # Subtraction of Handgrips (Left - Right)
        substring_to_remove = "session"
        df.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(L-R)"] = \
            df_left.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(left)"] - \
                df_right.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(right)"]

        return df

###############################################################################
    def calculate_laterality_index_hgs(self, df):
        """Calculate Laterality Index and add "hgs(LI)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for:
            HGS(Left - Right)/HGS(Left + Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        df_sub = self.calculate_sub_hgs(df)
        df_sum = self.calculate_sum_hgs(df)
        
        substring_to_remove = "session"
        df.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(LI)"] = \
            df_sub.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(L-R)"] / \
                df_sum.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}_hgs(L+R)"]


        return df
    
###############################################################################