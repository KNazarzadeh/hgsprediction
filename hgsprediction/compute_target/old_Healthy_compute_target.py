#!/usr/bin/env Disorderspredwp3

"""
Compute Target, Calculate and Add new columns based on corresponding Field-IDs and conditions

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

from ptpython.repl import embed

###############################################################################
class HealthyTargetComputing:
    def __init__(self, df, mri_status):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df
        self.mri_status = mri_status
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(mri_status, str), "df must be a string!"
        
        if mri_status == "nonmri":
            self.session = 0
        elif mri_status == "mri":
            self.session = 2

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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # ------------------------------------
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
        index = df[df.loc[:,f"1707-{session}.0"] == 1.0].index
        # Add and new column "hgs_dominant"
        # And assign Right hand HGS value:
        df.loc[index, f"hgs_dominant-{session}.0"] = \
            df.loc[:, f"47-{session}.0"]
        # ------------------------------------
        # If handedness is equal to 2
        # Left hand is Dominant
        # Find handedness equal to 2:
        index = df[df.loc[:,f"1707-{session}.0"] == 2.0].index
        # Add and new column "hgs_dominant"
        # And assign Left hand HGS value:
        df.loc[index, f"hgs_dominant-{session}.0"] = \
            df.loc[:, f"46-{session}.0"]
        # ------------------------------------
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer) OR
        # NaN value
        # Dominant will be the Highest Handgrip score from both hands.
        # Find handedness equal to 3, -3 or NaN:
        index = df[(df.loc[:, f"1707-{session}.0"] == 3.0) |
                (df.loc[:, f"1707-{session}.0"] == -3.0)|
                (df.loc[:, f"1707-{session}.0"].isna())].index
        if session == 0:
            # Add and new column "hgs_dominant"
            # And assign Highest HGS value among Right and Left HGS:        
            df.loc[index, f"hgs_dominant-{session}.0"] = \
                df.loc[:, [f"46-{session}.0", f"47-{session}.0"]].max(axis=1)
        elif session == 2:
            for idx in index:
                if df.loc[idx,"1707-0.0"] == 1.0:
                    # Add and new column "hgs_dominant"
                    # And assign Right hand HGS value:
                    df.loc[idx, f"hgs_dominant-{session}.0"] = \
                        df.loc[idx, "47-0.0"]
                elif df.loc[idx,"1707-0.0"] == 2.0:
                    # Add and new column "hgs_dominant"
                    # And assign Left hand HGS value:
                    df.loc[idx, f"hgs_dominant-{session}.0"] = \
                        df.loc[idx, "46-0.0"]
                else:
                     # Add and new column "hgs_dominant"
                    # And assign Highest HGS value among Right and Left HGS:        
                    df.loc[idx, f"hgs_dominant-{session}.0"] = \
                        df.loc[:, [f"46-{session}.0", f"47-{session}.0"]].max(axis=1)
                
        return df

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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # ------------------------------------
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
        # index = np.where(df.loc[:, f"1707-{session}.0"] == 1.0)[0]
        index = df[df.loc[:,f"1707-{session}.0"] == 1.0].index
        # Add and new column "hgs_nondominant"
        # And assign Left hand HGS value:
        df.loc[index, f"hgs_nondominant-{session}.0"] = \
            df.loc[:, f"46-{session}.0"]
        # ------------------------------------
        # If handedness is equal to 2
        # Right hand is Non-Dominant
        # Find handedness equal to 2:
        # index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        index = df[df.loc[:,f"1707-{session}.0"] == 2.0].index
        # Add and new column "hgs_nondominant"
        # And assign Right hand HGS value:        
        df.loc[index, f"hgs_nondominant-{session}.0"] = \
            df.loc[:, f"47-{session}.0"]
        # ------------------------------------
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer) OR
        # NaN value
        # Non-Dominant will be the Lowest Handgrip score from both hands.
        # Find handedness equal to 3, -3 or NaN:
        # index = np.where(
        #     (df.loc[:, f"1707-{session}.0"] == 3.0) |
        #     (df.loc[:, f"1707-{session}.0"] == -3.0) |
        #     (df.loc[:, f"1707-{session}.0"].isna()))[0]
        if session == 0:
            index = df[(df.loc[:, f"1707-{session}.0"] == 3.0) |
                    (df.loc[:, f"1707-{session}.0"] == -3.0)|
                    (df.loc[:, f"1707-{session}.0"].isna())].index
            # Add and new column "hgs_nondominant"
            # And assign Highest HGS value among Right and Left HGS:        
            df.loc[index, f"hgs_nondominant-{session}.0"] = \
                df.loc[:, [f"46-{session}.0", f"47-{session}.0"]].max(axis=1)
        elif session == 2:
            index = df[(df.loc[:, f"1707-{session}.0"] == 3.0) |
                    (df.loc[:, f"1707-{session}.0"] == -3.0)|
                    (df.loc[:, f"1707-{session}.0"].isna())].index
            for idx in index:
                if df.loc[idx,"1707-0.0"] == 1.0:
                    # Add and new column "hgs_nondominant"
                    # And assign Left hand HGS value:
                    df.loc[idx, f"hgs_nondominant-{session}.0"] = \
                        df.loc[idx, "46-0.0"]
                elif df.loc[idx,"1707-0.0"] == 2.0:
                    # Add and new column "hgs_nondominant"
                    # And assign Right hand HGS value:
                    df.loc[idx, f"hgs_nondominant-{session}.0"] = \
                        df.loc[idx, "47-0.0"]
                elif (df.loc[idx, "1707-0.0"] == 3.0) | \
                    (df.loc[idx, "1707-0.0"] == -3.0)| \
                    (df.loc[idx, "1707-0.0"].isna()):
                     # Add and new column "hgs_nondominant"
                    # And assign Highest HGS value among Right and Left HGS:        
                    df.loc[idx, f"hgs_nondominant-{session}.0"] = \
                        df.loc[idx, ["46-0.0", "47-0.0"]].max(axis=1) 
    
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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # ------------------------------------
        # Add new column "hgs(L+R)" by the following process: 
        # hgs_left field-ID: 46
        # hgs_right field-ID: 47
        # sum of Handgrips (Left + Right)
        df.loc[:, f"hgs(L+R)-{session}.0"] = \
            df.loc[:, f"46-{session}.0"] + df.loc[:, f"47-{session}.0"]

        return df

###############################################################################
    def calculate_left_hgs(self, df):
        """Calculate left and add "hgs(left)" column to dataframe

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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47

        df.loc[:, f"hgs(left)-{session}.0"] = df.loc[:, f"46-{session}.0"]

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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47

        df.loc[:, f"hgs(right)-{session}.0"] = df.loc[:, f"47-{session}.0"]

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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # ------------------------------------
        # Add new column "hgs(L-R)" by the following process: 
        # hgs_left field-ID: 46
        # hgs_right field-ID: 47
        # Subtraction of Handgrips (Left - Right)
        df.loc[:, f"hgs(L-R)-{session}.0"] = \
            df.loc[:, f"46-{session}.0"] - df.loc[:, f"47-{session}.0"]

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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47

        df_sub = self.calculate_sub_hgs(df)
        df_sum = self.calculate_sum_hgs(df)

        # Laterality Index of Handgrip strength: (Left - Right)/(Left +right)
        df.loc[:, f"hgs(LI)-{session}.0"] = \
            (df_sub.loc[:, f"hgs(L-R)-{session}.0"] / df_sum.loc[:, f"hgs(L+R)-{session}.0"])

        return df
###############################################################################
def compute_target(df, mri_status, target):
    if mri_status == "nonmri":
        session = 0
    elif mri_status == "mri":
        session = 2

    if target == "hgs_left":
        df.loc[:, f"hgs_left-{session}.0"] = df.loc[:, f"46-{session}.0"]
    elif target == "hgs_right":
        df.loc[:, f"hgs_right-{session}.0"] = df.loc[:, f"47-{session}.0"]        
        
    return df

###############################################################################
# Define the target which should be predict.
def extract_target(
    df,
    target,
):
    """
    Define target.

    Parameters
    ----------
    population: str
        Name of the population which to  to be analyse.

    Returns
    --------
    target : str
        List of different list of features.

    """
    filter_col = [col for col in df if col.startswith(f"{target}_hgs")]
    y = filter_col[0]

    return y
