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
    def remove_missing_hgs(self, df):
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
        index = df[((~df[f'{hgs_left}-{session}.0'].isna()) & (df[f'{hgs_left}-{session}.0'] != 0.0))
                   & ((~df[f'{hgs_right}-{session}.0'].isna()) & (df[f'{hgs_right}-{session}.0'] != 0.0))].index

        df = df.loc[index, :]

        return df
###############################################################################
    def define_handedness(self, df):
          
        # Extract columns "1707-0.0", "1707-1.0", "1707-2.0" for original_handedness
        original_handedness = df.loc[:, ["1707-0.0", "1707-1.0", "1707-2.0"]]
        
        # Find indices with NaN in the first column of original_handedness
        # Step 1: Identify rows where "1707-0.0" is NaN ,3 or -3
        index_unavailable = original_handedness[(original_handedness.loc[:, "1707-0.0"].isna())].index

        # Replace NaN in the first column with the max of the corresponding row
        original_handedness.loc[index_unavailable, "1707-0.0"] = np.nanmax(original_handedness.loc[index_unavailable, :], axis=1)           
        
        # Find indices where the first column equals -3 and set them to NaN
        index_no_answer = original_handedness[original_handedness.loc[:, "1707-0.0"] == -3].index
        original_handedness.loc[index_no_answer, "1707-0.0"] = np.nan
         
        # Remove all columns except the first and add it to df as new column
        df.loc[:, "original_handedness"] = original_handedness.loc[:, "1707-0.0"]
        
        # If handedness is equal to 1 --> Right hand is Dominant
        # Find handedness equal to left-handed, right-handed, and other
        index_right = df[df.loc[:, "original_handedness"] == 1].index
        index_left = df[df.loc[:, "original_handedness"] == 2].index                
        index_other = df[(df.loc[:, "original_handedness"] != 1) & (df.loc[:, "original_handedness"] != 2)].index

        df.loc[index_right, "handedness"] = 1.0
        df.loc[index_left, "handedness"] = 2.0

        if len(index_other) > 0:
            # Get the indices where the values in the two columns are equal    
            # Filter the DataFrame to include only the specified indexes
            filtered_df = df.loc[index_other]
                  
            # Find the indexes where the values in Column1 and Column2 are equal within the filtered DataFrame
            index_other_not_equal_hgs = filtered_df[filtered_df["47-0.0"] != filtered_df["46-0.0"]].index
            # Find the column with the maximum value among '46-0.0' and '47-0.0' for filtered rows
            result_column = df.loc[index_other_not_equal_hgs, ["47-0.0", "46-0.0"]].idxmax(axis=1)
            condition_right_index = result_column[result_column == "47-0.0"].index
            df.loc[condition_right_index, "handedness"] = 31.0
            condition_left_index = result_column[result_column == "46-0.0"].index
            df.loc[condition_left_index, "handedness"] = 32.0
            
            # Find the indexes where the values in Column1 and Column2 are equal within the filtered DataFrame
            index_other_equal_hgs = filtered_df[filtered_df["47-0.0"] == filtered_df["46-0.0"]].index
            # set 4.0 as ambidextrous
            df.loc[index_other_equal_hgs, "handedness"] = 4.0
            
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
        # df = df[(df.loc[:, hgs_dominant] >= 4) & (~df.loc[:, hgs_dominant].isna())]
        # df = df[(df.loc[:, hgs_nondominant] >= 4) & (~df.loc[:, hgs_nondominant].isna())]
        df = df[~df.loc[:, hgs_dominant].isna()]
        df = df[~df.loc[:, hgs_nondominant].isna()]        
        df = df[(df.loc[:, hgs_dominant] >= 4.0) & (df.loc[:, hgs_dominant] >= df.loc[:, hgs_nondominant])]

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

        # -----------------------------------------------------------
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
        # If handedness is equal to 1 --> Right hand is Dominant
        # Find handedness equal to left-handed, right-handed, and other
        index_right = df[df.loc[:, "handedness"] == 1.0].index
        index_left = df[df.loc[:, "handedness"] == 2.0].index  
        index_other_max_hgs_right = df[df.loc[:, "handedness"] == 31.0].index   
        index_other_max_hgs_left = df[df.loc[:, "handedness"] == 32.0].index                         
        index_other_equal_hgs = df[df.loc[:, "handedness"] == 4.0].index   
        # -----------------------------------------------------------    
        # -----------------------------------------------------------             
 
        df.loc[index_right, handedness] = 1.0
        df.loc[index_right, hgs_dominant] = df.loc[index_right, f"47-{session}.0"]
        df.loc[index_right, hgs_dominant_side] = "right"
        df.loc[index_right, hgs_nondominant] = df.loc[index_right, f"46-{session}.0"]
        df.loc[index_right, hgs_nondominant_side] = "left"
        
        df.loc[index_left, handedness] = 2.0
        df.loc[index_left, hgs_dominant] = df.loc[index_left, f"46-{session}.0"]
        df.loc[index_left, hgs_dominant_side] = "left"
        df.loc[index_left, hgs_nondominant] = df.loc[index_left, f"47-{session}.0"]
        df.loc[index_left, hgs_nondominant_side] = "right"

        df.loc[index_other_max_hgs_right, handedness] = 31.0
        df.loc[index_other_max_hgs_right, hgs_dominant] = df.loc[index_other_max_hgs_right, f"47-{session}.0"]
        df.loc[index_other_max_hgs_right, hgs_dominant_side] = "ambidextrous"
        df.loc[index_other_max_hgs_right, hgs_nondominant] = df.loc[index_other_max_hgs_right, f"46-{session}.0"]
        df.loc[index_other_max_hgs_right, hgs_nondominant_side] = "ambidextrous"
        
        df.loc[index_other_max_hgs_left, handedness] = 32.0
        df.loc[index_other_max_hgs_left, hgs_dominant] = df.loc[index_other_max_hgs_left, f"46-{session}.0"]
        df.loc[index_other_max_hgs_left, hgs_dominant_side] = "ambidextrous"
        df.loc[index_other_max_hgs_left, hgs_nondominant] = df.loc[index_other_max_hgs_left, f"47-{session}.0"]
        df.loc[index_other_max_hgs_left, hgs_nondominant_side] = "ambidextrous"
        
        df.loc[index_other_equal_hgs, handedness] = 4.0
        equal_hgs = df.loc[index_other_equal_hgs, f"47-{session}.0"]
        df.loc[index_other_equal_hgs, hgs_dominant] = equal_hgs
        df.loc[index_other_equal_hgs, hgs_dominant_side] = "ambidextrous"
        df.loc[index_other_equal_hgs, hgs_nondominant] = equal_hgs
        df.loc[index_other_equal_hgs, hgs_nondominant_side] = "ambidextrous"
        
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
