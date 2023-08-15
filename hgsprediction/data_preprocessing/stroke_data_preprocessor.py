#!/usr/bin/env Disorderspredwp3
"""Perform different preprocessing on stroke."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import pandas as pd
# from ptpython.repl import embed
# print('Done!')
# embed(globals(), locals())

from cmath import isnan
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from datetime import datetime as dt
from ptpython.repl import embed

###############################################################################
class StrokeMainDataPreprocessor:
    def __init__(self, df):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df

###############################################################################
    def remove_missing_stroke_dates(self, df):
        """ Drop all subjects who has no date of stroke and
            all dates of 1900-01-01 epresents "Date is unknown".

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/stroke.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.   
        """
        df = df[~df['42006-0.0'].isna() & df['42006-0.0'] != "1900-01-01"]

        return df
    
###############################################################################
    def remove_missing_hgs(self, df):
        
        # Handgrip strength info
        # for Left and Right Hands
        hgs_left = "46"  # Handgrip_strength_(left)
        hgs_right = "47"  # Handgrip_strength_(right)
        # UK Biobank assessed handgrip strength in 4 sessions
        session = 4 # 0 to 3
        
        df_output = pd.DataFrame()
        
        for ses in range(session):
            df_tmp = df[
                ((~df[f'{hgs_left}-{ses}.0'].isna()) &
                (df[f'{hgs_left}-{ses}.0'] !=  0))
                & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
                (df[f'{hgs_right}-{ses}.0'] !=  0))
            ]
            df_output = pd.concat([df_output, df_tmp])

        # Drop the duplicated subjects
        # based on 'eid' column (subject ID)
        df_output = df_output.drop_duplicates(keep="first")

        return df

###############################################################################
################################ Calculate Followup Days ######################
# The main goal of Followup Days calculation is to calculate the difference days 
# between stroke date and the attendence dates (for each session) 
# To see each subject had stroke before/after stroke
# to find the subject's stroke cohort (Pre-, Post- or Longitudinal)
###############################################################################
    def define_followup_days(self, df):
        """Calcuate the days differences between
            the Attendance date (the visit in clinic) and the Onset date of disease.
            
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        onset :  array
            The column of the disease onset date when the disease occurred.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        sessions = 4
        onset_date = pd.to_datetime(df['42006-0.0'])

        for ses in range(0, sessions):
            attendance_date = pd.to_datetime(df[f'53-{ses}.0'])
            df[f'followup_days-{ses}.0'] = (attendance_date-onset_date).dt.days

        return df

###############################################################################
    def define_stroke_type(self,df):

        stroke_subtypes = [
            '42008-0.0',	# Date of ischaemic stroke
            '42010-0.0',	# Date of intracerebral haemorrhage
            '42012-0.0',	# Date of subarachnoid haemorrhage
        ]
        df_subtype = df[stroke_subtypes]
        # Convert the columns to datetime
        df_subtype = df_subtype.apply(pd.to_datetime)

        # Find the earliest date on each row
        df['primary_subtype'] = df_subtype.min(axis=1)
        df['stroke_subtype_field'] = df_subtype.idxmin(axis=1)
        
        df.loc[df[df['stroke_subtype_field']=="42008-0.0"].index, 'stroke_subtype'] = "ischaemic"
        df.loc[df[df['stroke_subtype_field']=="420010-0.0"].index, 'stroke_subtype'] = "intracerebral_haemorrhage"
        df.loc[df[df['stroke_subtype_field']=="420012-0.0"].index, 'stroke_subtype'] = "subarachnoid_haemorrhage"

        return df
    
###############################################################################
    def extract_post_stroke_df(self, df):
        """Extract the post stroke dataframe from the stroke dataframe. 
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/stroke.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        # UKB contains 4 assessment sessions
        sessions = 4
        # Initialize an empty list
        followupdays_cols = []
        # Append "followup_days" column names to list
        for ses in range(0, sessions):
            followupdays_cols.append(f"followup_days-{ses}.0")

        # Create a boolean mask for rows where all numeric variables are greater than zero or are NaN
        mask = df[followupdays_cols].apply(lambda row: np.all(np.logical_or(np.isnan(row), row >= 0)), axis=1)

        # Apply the mask to the DataFrame to get the desired rows
        # for only Post-stroke subjects
        post_stroke_df = df[mask]

        return post_stroke_df

###############################################################################
    def extract_pre_stroke_df(self, df):
        """Extract the pre stroke dataframe from the stroke dataframe. 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/stroke.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        sessions = 4
        followupdays_cols = []
        for ses in range(0, sessions):
            followupdays_cols.append(f"followup_days-{ses}.0")

        # Create a boolean mask for rows where all numeric variables are lower than zero or are NaN
        mask = df[followupdays_cols].apply(lambda row: np.all(np.logical_or(np.isnan(row), row < 0)), axis=1)

        # Apply the mask to the DataFrame to get the desired rows
        # for only Pre-stroke subjects
        post_stroke_df = df[mask]

        return post_stroke_df

###############################################################################
    def extract_longitudinal_stroke_df(self, df):
        """Extract the longitudinal dataframe from the stroke dataframe. 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/stroke.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        post_stroke_df = self.extract_post_stroke_df(df)
        pre_stroke_df = self.extract_pre_stroke_df(df)

        # The intersection between pre and post dataframes of stroke
        # will be the longitudinal dataframe.
        # Merge the two DataFrames
        merged_df = pd.concat([pre_stroke_df, post_stroke_df]).drop_duplicates()

        # Find the symmetric difference between
        # the merged pre- and post- stroke DataFrames and the original DataFrame
        longitudinal_stroke_df = df.merge(merged_df, how='outer', indicator=True).query('_merge != "both"').drop('_merge', axis=1)

        return longitudinal_stroke_df

###############################################################################
###############################################################################
    def extract_all_post_stroke_visits(self, df):
        """Extract the post stroke dataframe from the stroke dataframe. 
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/stroke.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        # Assign corresponding session number from the Class:
        stroke_cohort = "post-stroke"
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"

        sessions = 4
        followupdays_cols = []
        for ses in range(0, sessions):
            followupdays_cols.append(f"followup_days-{ses}.0")
        # Drop rows where values in all four columns of followup_days 
        # are less than 0 (Drop All Pre-stroke subjects)
        # And keep rows with at least one value (>= 0)
        filtered_df = df.loc[(df[followupdays_cols] >= 0).any(axis=1), followupdays_cols]
        # Apply a lambda function to each element in the DataFrame 'filtered_df'.
        # The lambda function replaces negative values with NaN (Not a Number),
        # while leaving non-negative values unchanged.
        cleaned_df = filtered_df.applymap(lambda x: np.nan if x < 0 else x)
        # Function to sort each row and return column names, handling NaN values
        # Sorts the values in a row in ascending order while placing NaN values at the end,
        # and returns a list of column names where non-NaN values are preserved in the sorted order.
        # Any NaN values are replaced with np.nan in the resulting list.   
        def sort_and_get_columns(row):
            sorted_row = row.sort_values(na_position='last')  # Sort the row values, keeping NaNs at the end.
            result = []
            for col in sorted_row.index:
                if not pd.isna(sorted_row[col]):  # Check if the value is not NaN.
                    result.append(col)  # Append the column name to the result list.
                else:
                    result.append(np.nan)  # Append np.nan if the value is NaN.
            return result  # Return the list of column names preserving the sorted non-NaN values.
            # Another way to write code --> return [col if not pd.isna(sorted_row[col]) else np.nan for col in sorted_row.index]

        # Apply the 'sort_and_get_columns' function to each row of the cleaned DataFrame ('cleaned_df').
        # The function sorts row values in ascending order with NaNs placed at the end,
        # and returns a DataFrame where each row contains the sorted column names with NaN values replaced by np.nan.
        sorted_columns = cleaned_df.apply(sort_and_get_columns, axis=1, result_type='expand')
        
        # Set custom column names for the sorted_columns DataFrame
        sorted_columns.columns = [f"1st_{stroke_cohort}-stroke_session",
                                  f"2nd_{stroke_cohort}-stroke_session",
                                  f"3rd_{stroke_cohort}-stroke_session",
                                  f"4th_{stroke_cohort}-stroke_session"]
        # Function to remove a substring
        def remove_substring(value, substring):
            if isinstance(value, str):
                return value.replace(substring, '')
            return value
        substring_to_remove = "followup_days-"
        # Apply the function to remove the substring from all values
        post_stroke_visits_df = sorted_columns.applymap(lambda x: remove_substring(x, substring_to_remove))

        return post_stroke_visits_df

###############################################################################
def extract_all_pre_stroke_visits(self, df):
        """Extract the pre stroke dataframe from the stroke dataframe. 
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/stroke.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        # Assign corresponding session number from the Class:
        stroke_cohort = "pre-stroke"
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"

        sessions = 4
        followupdays_cols = []
        for ses in range(0, sessions):
            followupdays_cols.append(f"followup_days-{ses}.0")
        # Drop rows where values in all four columns of followup_days 
        # are greather or equal to 0 (Drop All Post-stroke subjects)
        # And keep rows with at least one value (< 0)
        filtered_df = df.loc[(df[followupdays_cols] < 0).any(axis=1), followupdays_cols]
        
        # Sort column names based on values if values < threshold for each row
        # columns with values lower than 0 is all pre-stroke
        sorted_columns = filtered_df.apply(lambda row: sorted([col for col in row.index if row[col] < 0]),
                          axis=1, result_type='expand')
        # Set custom column names for the sorted_columns DataFrame
        sorted_columns.columns = [f"1st_{stroke_cohort}-stroke_session",
                                  f"2nd_{stroke_cohort}-stroke_session",
                                  f"3rd_{stroke_cohort}-stroke_session",
                                  f"4th_{stroke_cohort}-stroke_session"]
        # Function to remove a substring
        def remove_substring(value, substring):
            if isinstance(value, str):
                return value.replace(substring, '')
            return value
        substring_to_remove = "followup_days-"
        # Apply the function to remove the substring from all values
        pre_stroke_visits_df = sorted_columns.applymap(lambda x: remove_substring(x, substring_to_remove))

        return pre_stroke_visits_df
###############################################################################
class StrokeExtraDataPreprocessor:
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

################################ DATA VALIDATION ##############################
# The main goal of data validation is to verify that the data is 
# accurate, reliable, and suitable for the intended analysis.
###############################################################################
    def validate_handgrips(self, df):
        """Exclude all subjects who had Dominant HGS < 4:

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
        """        
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Calculate Dominant and Non-Dominant HGS by
        # Calling the modules:
        df = self.calculate_dominant_hgs(df)
        # ------------------------------------
        # Exclude all subjects who had Dominant HGS < 4:
        # The condition is applied to "hgs_dominant" columns
        # And then reset_index the new dataframe:
        df = df[df.loc[:, f"{session_column[:, len(substring_to_remove)].strip()}hgs_dominant"] >=4]

        return df

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
                df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}hgs_dominant"] = \
                    df.loc[idx, f"47-{session}"]
            # ------------------------------------
            # If handedness is equal to 2
            # Right hand is Non-Dominant
            # Find handedness equal to 2:
            elif df.loc[idx,f"1707-{session}"] == 2.0:
                # Add and new column "hgs_dominant"
                # And assign Left hand HGS value:        
                df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}hgs_dominant"] = \
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
                    df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}hgs_dominant"] = \
                        df.loc[idx, [f"46-{session}", f"47-{session}"]].max(axis=1)
            elif session == 2:
                if (df.loc[idx, f"1707-{session}"] == 3.0) | \
                    (df.loc[idx, f"1707-{session}"] == -3.0) | \
                    (df.loc[idx, f"1707-{session}"].isna()):
                    if df.loc[idx,"1707-0.0"] == 1.0:
                        # Add and new column "hgs_dominant"
                        # And assign Right hand HGS value:
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}hgs_dominant"] = \
                            df.loc[idx, "47-0.0"]
                    elif df.loc[idx,"1707-0.0"] == 2.0:
                        # Add and new column "hgs_dominant"
                        # And assign Left hand HGS value:
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}hgs_dominant"] = \
                            df.loc[idx, "46-0.0"]
                            
                    elif (df.loc[idx, "1707-0.0"] == 3.0) | \
                        (df.loc[idx, "1707-0.0"] == -3.0)| \
                        (df.loc[idx, "1707-0.0"].isna()):
                        # Add and new column "hgs_dominant"
                        # And assign Highest HGS value among Right and Left HGS:        
                        df.loc[idx, f"{session_column[:, len(substring_to_remove)].strip()}hgs_dominant"] = \
                            df.loc[idx, ["46-0.0", "47-0.0"]].max(axis=1) 

