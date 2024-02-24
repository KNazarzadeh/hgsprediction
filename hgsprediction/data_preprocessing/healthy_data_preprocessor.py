#!/usr/bin/env Disorderspredwp3

"""
Preprocess data, Calculate and Add new columns based on corresponding Field-IDs,
conditions to data

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

from ptpython.repl import embed

###############################################################################
class HealthyDataPreprocessor:
    def __init__(self, df, mri_status, session):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        mri_status == "nonmri":
        #     self.session = "0"
        # elif mri_status == "mri":
        #     self.session = "2"
        """
        self.df = df
        self.mri_status = mri_status
        self.session = session
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(mri_status, str), "df must be a string!"
        

############################# CHECK HGS AVAILABILITY ##########################
# The main goal of check HGS availability is to check if the right and left HGS
# be available for each session.
###############################################################################
    def check_hgs_availability(self, df):
        """Check HGS availability on right and left for each session:

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
        """
        # Assign corresponding session number from the Class:
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, str), "session must be a string!"
        # Handgrip strength info
        # for Left and Right Hands
        hgs_left = "46"  # Handgrip_strength_(left)
        hgs_right = "47"  # Handgrip_strength_(right)
        # UK Biobank assessed handgrip strength in 4 sessions
        index = df[((~df[f'{hgs_left}-{session}.0'].isna()) &
                (df[f'{hgs_left}-{session}.0'] !=  0) & (df[f'{hgs_left}-{session}.0'] >= 4.0))
                & ((~df[f'{hgs_right}-{session}.0'].isna()) &
                (df[f'{hgs_right}-{session}.0'] !=  0) & (df[f'{hgs_right}-{session}.0'] >= 4.0))].index

        df = df.loc[index, :]

        return df
################################ DATA VALIDATION ##############################
# The main goal of data validation is to verify that the data is 
# accurate, reliable, and suitable for the intended analysis.
###############################################################################
    def validate_handgrips(self, df):
        """Exclude all subjects who had Dominant HGS < 4 and != NaN:

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
        """
        # Assign corresponding session number from the Class:
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, str), "session must be a string!"
        # ------------------------------------
        # Calculate Dominant and Non-Dominant HGS by
        # Calling the modules:
        df = self.calculate_dominant_nondominant_hgs(df)
        hgs_dominant = f"hgs_dominant-{session}.0"
        hgs_nondominant = f"hgs_nondominant-{session}.0"
        # ------------------------------------
        # Exclude all subjects who had Dominant HGS < 4:
        # The condition is applied to "hgs_dominant" columns
        # And then reset_index the new dataframe:
        df = df[(df.loc[:, hgs_dominant] >= 4) & (~df.loc[:, hgs_dominant].isna())]
        df = df[(df.loc[:, hgs_nondominant] >= 4) & (~df.loc[:, hgs_nondominant].isna())]
        df = df[(df.loc[:, hgs_dominant] >= df.loc[:, hgs_nondominant])]

        return df

###############################################################################
    def calculate_dominant_nondominant_hgs(self, df):
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
        session = self.session
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, str), "session must be a string!"
        # -----------------------------------------------------------
        
        # Add a new column 'new_column'
        hgs_dominant = f"hgs_dominant-{session}.0"
        hgs_nondominant = f"hgs_nondominant-{session}.0"
        hgs_dominant_side = f"hgs_dominant_side-{session}.0"
        hgs_nondominant_side = f"hgs_nondominant_side-{session}.0"
        handedness = f"handedness-{session}.0"    

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
        if session in ["0", "1", "3"]:
            # Add and new column "hgs_dominant"
            # And assign Right hand HGS value
            idx = df[df.loc[:, "handness"] == 1.0].index
            df.loc[idx, handedness] = 1.0
            df.loc[idx, hgs_dominant] = df.loc[idx, "47-0.0"]
            df.loc[idx, hgs_dominant_side] = "right"              
            df.loc[idx, hgs_nondominant] = df.loc[idx, "46-0.0"]
            df.loc[idx, hgs_nondominant_side] = "left"
            # If handedness is equal to 2
            # Right hand is Non-Dominant
            # Find handedness equal to 2:
            # Add and new column "hgs_dominant"
            # And assign Left hand HGS value:
            idx = df[df.loc[:, "handness"] == 2.0].index
            df.loc[idx, handedness] = 2.0
            df.loc[idx, hgs_dominant] = df.loc[idx, "46-0.0"]
            df.loc[idx, hgs_dominant_side] = "left"              
            df.loc[idx, hgs_nondominant] = df.loc[idx, "47-0.0"]
            df.loc[idx, hgs_nondominant_side] = "right"
            
            # ------------------------------------
            # If handedness is equal to:
            # 3 (Use both right and left hands equally) OR
            # -3 (handiness is not available/Prefer not to answer) OR
            # NaN value
            # Dominant will be the Highest Handgrip score from both hands.
            # Find handedness equal to 3, -3 or NaN:
            # Add and new column "hgs_dominant"
            # And assign Highest HGS value among Right and Left HGS:
            # Add and new column "hgs_dominant"
            # And assign lowest HGS value among Right and Left HGS:
            idx = df[df.loc[:, "handness"].isin([3.0, -3.0, np.NaN])].index
            df.loc[idx, handedness] = 3.0
            df_tmp = df.loc[idx, :]
            idx_tmp = df_tmp[df_tmp.loc[:, "47-0.0"] == df_tmp.loc[:, "46-0.0"]].index
            df.loc[idx_tmp, hgs_dominant] = df.loc[idx, "47-0.0"]
            df.loc[idx_tmp, hgs_dominant_side] = "balanced_hgs"              
            df.loc[idx_tmp, hgs_nondominant] = df.loc[idx, "46-0.0"]
            df.loc[idx_tmp, hgs_nondominant_side] = "balanced_hgs"

            idx_tmp = df_tmp[df_tmp.loc[:, "47-0.0"] != df_tmp.loc[:, "46-0.0"]].index
            result_column = df.loc[idx_tmp, ["46-0.0", "47-0.0"]].idxmax(axis=1)
            condition_right = result_column[result_column =='47-0.0']
            df.loc[condition_right.index, hgs_dominant] = df.loc[condition_right.index, "47-0.0"]
            df.loc[condition_right.index, hgs_dominant_side] = "right"
            df.loc[condition_right.index, hgs_nondominant] = df.loc[condition_right.index, "46-0.0"]
            df.loc[condition_right.index, hgs_nondominant_side] = "left"
            condition_left = result_column[result_column =='46-0.0']
            df.loc[condition_left.index, hgs_dominant] = df.loc[condition_left.index, "46-0.0"]
            df.loc[condition_left.index, hgs_dominant_side] = "left"
            df.loc[condition_left.index, hgs_nondominant] = df.loc[condition_left.index, "47-0.0"]
            df.loc[condition_left.index, hgs_nondominant_side] = "right"

        elif session == "2":
            idx = df[df.loc[:, "handness"] == 1.0].index
            df.loc[idx, handedness] = 1.0
            df.loc[idx, hgs_dominant] = df.loc[idx, "47-2.0"]
            df.loc[idx, hgs_dominant_side] = "right"
            df.loc[idx, hgs_nondominant] = df.loc[idx, "46-2.0"]
            df.loc[idx, hgs_nondominant_side] = "left"
            
            idx = df[df.loc[:, "handness"] == 2.0].index
            df.loc[idx, handedness] = 2.0
            df.loc[idx, hgs_dominant] = df.loc[idx, "46-2.0"]
            df.loc[idx, hgs_dominant_side] = "left"             
            df.loc[idx, hgs_nondominant] = df.loc[idx, "47-2.0"]
            df.loc[idx, hgs_nondominant_side] = "right"
            # ------------------------------------
            # If handedness is equal to:
            # 3 (Use both right and left hands equally) OR
            # -3 (handiness is not available/Prefer not to answer) OR
            # NaN value
            # Dominant will be the Highest Handgrip score from both hands.
            # Find handedness equal to 3, -3 or NaN:
            # Add and new column "hgs_dominant"
            # And assign Highest HGS value among Right and Left HGS:
            # Add and new column "hgs_dominant"
            # And assign lowest HGS value among Right and Left HGS:
            idx = df[df.loc[:, "handness"].isin([3.0, -3.0, np.NaN])].index
            df.loc[idx, handedness] = 3.0
            df_tmp = df.loc[idx, :]
            idx_tmp = df_tmp[df_tmp.loc[:, "47-2.0"] == df_tmp.loc[:, "46-2.0"]].index
            df.loc[idx_tmp, hgs_dominant] = df.loc[idx, "47-2.0"]
            df.loc[idx_tmp, hgs_dominant_side] = "balanced_hgs"              
            df.loc[idx_tmp, hgs_nondominant] = df.loc[idx, "46-2.0"]
            df.loc[idx_tmp, hgs_nondominant_side] = "balanced_hgs"

            idx_tmp = df_tmp[df_tmp.loc[:, "47-2.0"] != df_tmp.loc[:, "46-2.0"]].index
            result_column = df.loc[idx_tmp, ["46-2.0", "47-2.0"]].idxmax(axis=1)
            condition_right = result_column[result_column =='47-2.0']
            df.loc[condition_right.index, hgs_dominant] = df.loc[condition_right.index, "47-2.0"]
            df.loc[condition_right.index, hgs_dominant_side] = "right"
            df.loc[condition_right.index, hgs_nondominant] = df.loc[condition_right.index, "46-2.0"]
            df.loc[condition_right.index, hgs_nondominant_side] = "left"
            condition_left = result_column[result_column =='46-2.0']
            df.loc[condition_left.index, hgs_dominant] = df.loc[condition_left.index, "46-2.0"]
            df.loc[condition_left.index, hgs_dominant_side] = "left"
            df.loc[condition_left.index, hgs_nondominant] = df.loc[condition_left.index, "47-2.0"]
            df.loc[condition_left.index, hgs_nondominant_side] = "right"
            # df_tmp = df.loc[idx, :]
            # idx_tmp = df_tmp[df_tmp.loc[:, "1707-0.0"] == 1.0].index
            # df.loc[idx_tmp, handedness] = 1.0
            # df.loc[idx_tmp, hgs_dominant] = df.loc[idx_tmp, "47-2.0"]
            # df.loc[idx_tmp, hgs_dominant_side] = "right"            
            # df.loc[idx_tmp, hgs_nondominant] = df.loc[idx_tmp, "46-2.0"]
            # df.loc[idx_tmp, hgs_nondominant_side] = "left"
            
            # idx_tmp = df_tmp[df_tmp.loc[:, "1707-0.0"] == 2.0].index
            # df.loc[idx_tmp, handedness] = 2.0            
            # df.loc[idx_tmp, hgs_dominant] = df.loc[idx_tmp, "46-2.0"]
            # df.loc[idx_tmp, hgs_dominant_side] = "left"            
            # df.loc[idx_tmp, hgs_nondominant] = df.loc[idx_tmp, "47-2.0"]
            # df.loc[idx_tmp, hgs_nondominant_side] = "right"
            
            # idx_tmp = df_tmp[df_tmp.loc[:, "1707-0.0"].isin([3.0, -3.0, np.NaN])].index
            # df.loc[idx_tmp, handedness] = 3.0
            # df_tmp_2 = df_tmp.loc[idx_tmp, :]
            # idx_tmp_2 = df_tmp_2[df_tmp_2.loc[:, "47-2.0"] == df_tmp_2.loc[:, "46-2.0"]].index
            # df.loc[idx_tmp_2, hgs_dominant] = df.loc[idx_tmp_2, "47-2.0"]
            # df.loc[idx_tmp_2, hgs_dominant_side] = "right"              
            # df.loc[idx_tmp_2, hgs_nondominant] = df.loc[idx_tmp_2, "46-2.0"]
            # df.loc[idx_tmp_2, hgs_nondominant_side] = "left"
            
            # idx_tmp_2 = df_tmp_2[df_tmp_2.loc[:, "47-2.0"] != df_tmp_2.loc[:, "46-2.0"]].index
            # result_column = df.loc[idx_tmp_2, ["46-2.0", "47-2.0"]].idxmax(axis=1)
            # condition_right = result_column[result_column =='47-2.0']
            # df.loc[condition_right.index, hgs_dominant] = df.loc[condition_right.index, "47-2.0"]
            # df.loc[condition_right.index, hgs_dominant_side] = "right"
            # df.loc[condition_right.index, hgs_nondominant] = df.loc[condition_right.index, "46-2.0"]
            # df.loc[condition_right.index, hgs_nondominant_side] = "left"
            # condition_left = result_column[result_column =='46-2.0']
            # df.loc[condition_left.index, hgs_dominant] = df.loc[condition_left.index, "46-2.0"]
            # df.loc[condition_left.index, hgs_dominant_side] = "left"
            # df.loc[condition_left.index, hgs_nondominant] = df.loc[condition_left.index, "47-2.0"]
            # df.loc[condition_left.index, hgs_nondominant_side] = "right"
            
        df.loc[:, "percent_diff_bw_dominant_nonominant"] = ((df.loc[:, hgs_dominant]-df.loc[:, hgs_nondominant])/df.loc[:, hgs_dominant])*100
        
        return df

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
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"

        nan_cols = df.columns[df.isna().all()].tolist()
        df = df.drop(nan_cols, axis=1)
        
        return df
    
###############################################################################
