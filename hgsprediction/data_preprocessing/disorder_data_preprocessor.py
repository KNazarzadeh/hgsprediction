#!/usr/bin/env Disorderspredwp3
"""Perform different preprocessing on disorder."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>
import pandas as pd
import numpy as np
from datetime import datetime as dt
from ptpython.repl import embed

# Add the warning suppression code here
###############################################################################
class DisorderMainDataPreprocessor:
    def __init__(self, df, disorder):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df
        self.disorder = disorder
###############################################################################
    def remove_missing_disorder_dates(self, df, disorder):
        """ Drop all subjects who has no date of disorder and
            all dates of 1900-01-01 epresents "Date is unknown".

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disorder.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.   
        """
        if disorder == "stroke":
            df = df[(~df.loc[:, "42006-0.0"].isna()) & (df.loc[:, "42006-0.0"] != "1900-01-01")]
            
        elif disorder == "parkinson":
            df = df[(~df.loc[:, "131022-0.0"].isna()) & (df.loc[:, "131022-0.0"] != "1900-01-01")]
            
        elif disorder == "depression":
            ## Convert both columns to datetime
            df.loc[:, "130894-0.0"] = pd.to_datetime(df.loc[:, "130894-0.0"])
            df.loc[:, "130896-0.0"] = pd.to_datetime(df.loc[:, "130896-0.0"])
            
            # Compare date of F32 and F33
            df["depression_onset"] = df.loc[:, ["130894-0.0", "130896-0.0"]].min(axis=1)

            df = df[(~df.loc[:, "depression_onset"].isna()) & (~df.loc[:, "depression_onset"].isin(["1900-01-01", "1901-01-01", "1902-02-02", "1903-03-03", "1909-09-09", "2037-07-07"]))]

        return df
    
###############################################################################
    def define_handness(self, df):
        
        handness = df.loc[:, ["1707-0.0", "1707-1.0", "1707-2.0"]]
        # Replace NaN values in the first column with maximum of respective row
        index_NaN = handness[handness.loc[:, "1707-0.0"].isna()].index
        max_value = handness.loc[index_NaN, ["1707-1.0", "1707-2.0"]].max(axis=1)
        handness.loc[index_NaN, "1707-0.0"] = max_value

        # Replace occurrences of -3 in the first column with NaN
        index_no_answer = handness[handness.loc[:, "1707-0.0"] == -3.0].index
        handness.loc[index_no_answer, "1707-0.0"] = np.NaN
        
        # Remove all columns except the first one
        df.loc[:, "handness"] = handness.loc[:, "1707-0.0"]
        
        return df
###############################################################################
    def remove_missing_hgs(self, df):
        
        # Handgrip strength info
        # for Left and Right Hands
        hgs_left = "46"  # Handgrip_strength_(left)
        hgs_right = "47"  # Handgrip_strength_(right)
        # UK Biobank assessed handgrip strength in 4 sessions
        sessions = 4 # 0 to 3
        
        df_output = pd.DataFrame()
        
        for ses in range(sessions):
            # df_tmp = df[
            #     ((~df.loc[:, f"{hgs_left}-{ses}.0"].isna()) &
            #     (df.loc[:, f"{hgs_left}-{ses}.0"] !=  0) & (df.loc[:, f"{hgs_left}-{ses}.0"] >= 4.0))
            #     & ((~df.loc[:, f"{hgs_right}-{ses}.0"].isna()) &
            #     (df.loc[:, f"{hgs_right}-{ses}.0"] !=  0) & (df.loc[:, f"{hgs_right}-{ses}.0"] >= 4.0))
            # ]
            df_tmp = df[(~df.loc[:, f"{hgs_left}-{ses}.0"].isna()) & ((~df.loc[:, f"{hgs_right}-{ses}.0"].isna()))]
            df_output = pd.concat([df_output, df_tmp], axis=0)

        # Drop the duplicated subjects
        # based on 'eid' column (subject ID)
        df_output = df_output.drop_duplicates(keep="first")

        return df_output

###############################################################################
################################ Calculate Followup Days ######################
# The main goal of Followup Days calculation is to calculate the difference days 
# between disorder date and the attendence dates (for each session) 
# To see each subject had disorder before/after disorder
# to find the subject's disorder cohort (Pre-, Post- or Longitudinal)
###############################################################################
    def define_followup_days(self, df, disorder):
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
        if disorder == "stroke":
            onset_date = pd.to_datetime(df.loc[:, "42006-0.0"])
            
        elif disorder == "parkinson":
            onset_date = pd.to_datetime(df.loc[:, "131022-0.0"])
            
        elif disorder == "depression":
            onset_date = pd.to_datetime(df.loc[:, "depression_onset"])

        for ses in range(0, sessions):
            attendance_date = pd.to_datetime(df.loc[:, f"53-{ses}.0"])
            df[f'followup_days-{ses}.0'] = (attendance_date-onset_date).dt.days

        return df

###############################################################################
    def define_disorder_type(self,df, disorder):

        if disorder == "stroke":
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
    def preprocess_disorder_df(self, df, disorder):
        """Use the disorder dataframe. 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disorder.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        sessions = 4
        followupdays_cols = []
        for ses in range(0, sessions):
            followupdays_cols.append(f"followup_days-{ses}.0")
        substring_to_remove = "followup_days-"
        # Function to remove a substring
        def remove_substring(value, substring):
            if isinstance(value, str):
                return value.replace(substring, '')
            return value
        ##################################################################
        ############### Extract All Post disorder Subjects #################
        ##################################################################
        disorder_cohort = f"post-{disorder}"
        
        # Drop rows where values in all four columns of followup_days 
        # are less than 0 (Drop All Pre-disorder subjects)
        # And keep rows with at least one value (>= 0)
        filtered_post_df = df.loc[(df[followupdays_cols] >= 0).any(axis=1), followupdays_cols]
        # Apply a lambda function to each element in the DataFrame 'filtered_df'.
        # The lambda function replaces negative values with NaN (Not a Number),
        # while leaving non-negative values unchanged.
        cleaned_post_df = filtered_post_df.applymap(lambda x: np.nan if x < 0 else x)
        # Function to sort each row and return column names, handling NaN values
        # Sorts the values in a row in ascending order while placing NaN values at the end,
        # and returns a list of column names where non-NaN values are preserved in the sorted order.
        # Any NaN values are replaced with np.nan in the resulting list.   
        def sort_and_get_post_visit_columns(row):
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
        post_sorted_columns = cleaned_post_df.apply(sort_and_get_post_visit_columns, axis=1, result_type='expand')
        # Set custom column names for the sorted_columns DataFrame
        post_sorted_columns.columns = [f"1st_{disorder_cohort}_session",
                                        f"2nd_{disorder_cohort}_session",
                                        f"3rd_{disorder_cohort}_session",
                                        f"4th_{disorder_cohort}_session"]
        
        # Apply the function to remove the substring from all values
        # And convert string to float
        post_sorted_columns = post_sorted_columns.applymap(lambda x: float(remove_substring(x, substring_to_remove)))
               
        # Combine selected rows from 'df' with sorted columns from 'sorted_columns' to create 'post_disorder_visits_df'
        post_disorder_visits_df = pd.concat([df[df.index.isin(post_sorted_columns.index)], post_sorted_columns], axis=1)
        ##################################################################        
        ############### Extract All Pre disorder Subjects #################
        ##################################################################
        disorder_cohort = f"pre-{disorder}"
        # Drop rows where values in all four columns of followup_days 
        # are greather than 0 (Drop All Post-disorder subjects)
        # And keep rows with at least one value (< 0)
        filtered_pre_df = df.loc[(df[followupdays_cols] < 0).any(axis=1), followupdays_cols]
        # Apply a lambda function to each element in the DataFrame 'filtered_df'.
        # The lambda function replaces positive/zero values with NaN (Not a Number),
        # while leaving negative values unchanged.
        cleaned_pre_df = filtered_pre_df.applymap(lambda x: np.nan if x >= 0 else x)
        # Function to sort each row and return column names, handling NaN values
        # Sorts the values in a row in descending order while placing NaN values at the end,
        # and returns a list of column names where non-NaN values are preserved in the sorted order.
        # Any NaN values are replaced with np.nan in the resulting list.   
        def sort_and_get_pre_visit_columns(row):
            sorted_row = row.sort_values(na_position='last', ascending=False)  # Sort descending row values, keeping NaNs at the end.
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
        pre_sorted_columns = cleaned_pre_df.apply(sort_and_get_pre_visit_columns, axis=1, result_type='expand')
        
        # Set custom column names for the sorted_columns DataFrame
        pre_sorted_columns.columns = [f"1st_{disorder_cohort}_session",
                                        f"2nd_{disorder_cohort}_session",
                                        f"3rd_{disorder_cohort}_session",
                                        f"4th_{disorder_cohort}_session"]

        # Apply the function to remove the substring from all values
        # And convert string to float
        pre_sorted_columns = pre_sorted_columns.applymap(lambda x: float(remove_substring(x, substring_to_remove)))
        # Combine selected rows from 'df' with sorted columns from 'sorted_columns' to create 'pre_disorder_visits_df'
        pre_disorder_visits_df = pd.concat([df[df.index.isin(pre_sorted_columns.index)], pre_sorted_columns], axis=1)
        
        ##################################################################        
        ############### Extract Longitudinal disorder Subjects #############
        ##################################################################
        # The intersection between pre and post dataframes of disorder
        # will be the longitudinal dataframe.
        # Find duplicate index values
        duplicate_index = pre_disorder_visits_df.index.intersection(post_disorder_visits_df.index)
        # Find duplicate index values on the original DataFrame
        df_longitudinal_disorder = df[df.index.isin(duplicate_index)]
        # List of new pre_visits column names
        pre_visits_columns = [f"1st_pre-{disorder}_session",
                              f"2nd_pre-{disorder}_session",
                              f"3rd_pre-{disorder}_session",
                              f"4th_pre-{disorder}_session"]
        # List of new post_visits column names
        post_visits_columns = [f"1st_post-{disorder}_session",
                               f"2nd_post-{disorder}_session",
                               f"3rd_post-{disorder}_session",
                               f"4th_post-{disorder}_session"]

        # Join pre_visits columns from df_all_pre_disorder_visits into longitudinal_disorder_df    
        df_longitudinal_disorder = df_longitudinal_disorder.join(pre_disorder_visits_df[pre_disorder_visits_df.index.isin(duplicate_index)][pre_visits_columns])
        # Join post_visits columns from df_all_post_disorder_visits into longitudinal_disorder_df
        df_longitudinal_disorder = df_longitudinal_disorder.join(post_disorder_visits_df[post_disorder_visits_df.index.isin(duplicate_index)][post_visits_columns])
        ##################################################################        
        ############### Merge All 3 types of sub-type disorder #############
        ##################################################################
        # Merge All 3 types of sub-type disorder dataframes 
        # while excluding duplicates from pre-disorder and post-disorder visits,
        # and include the longitudinal disorder dataframe.
        merged_df = pd.concat([pre_disorder_visits_df[~pre_disorder_visits_df.index.isin(duplicate_index)],
                               post_disorder_visits_df[~post_disorder_visits_df.index.isin(duplicate_index)],
                               df_longitudinal_disorder])
        # Reindex the merged dataframe to align with the original index of df,
        # ensuring consistent indexing for further analysis or operations.
        preprocessed_df = merged_df.reindex(df.index)
                
        return preprocessed_df

###############################################################################
    def calculate_dominant_nondominant_hgs(self, df, disorder):
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
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        for disorder_cohort in [f"pre-{disorder}", f"post-{disorder}"]:
            for visit_session in range(1, 4):
                if visit_session == 1:
                    session_column = f"1st_{disorder_cohort}_session"
                elif visit_session == 2:
                    session_column = f"2nd_{disorder_cohort}_session"
                elif visit_session == 3:
                    session_column = f"3rd_{disorder_cohort}_session"
                elif visit_session == 4:
                    session_column = f"4th_{disorder_cohort}_session" 
                # Add a new column 'new_column'
                hgs_dominant = session_column.replace(substring_to_remove, "hgs_dominant")
                hgs_nondominant = session_column.replace(substring_to_remove, "hgs_nondominant")
                hgs_dominant_side = session_column.replace(substring_to_remove, "hgs_dominant_side")
                hgs_nondominant_side = session_column.replace(substring_to_remove, "hgs_nondominant_side")
                handedness = session_column.replace(substring_to_remove, "handedness")    

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
                # If handedness is equal to 1
                # Right hand is Dominant
                # Find handedness equal to 1:
                for ses in range(4):
                    index = df[df.loc[:, session_column] == ses].index
                    filtered_df = df.loc[index, :]
                    idx = filtered_df[filtered_df.loc[:, "handness"] == 1.0].index
                    df.loc[idx, handedness] = 1.0
                    df.loc[idx, hgs_dominant] = df.loc[idx, f"47-{ses}.0"]
                    df.loc[idx, hgs_dominant_side] = "right"
                    df.loc[idx, hgs_nondominant] = df.loc[idx, f"46-{ses}.0"]
                    df.loc[idx, hgs_nondominant_side] = "left"

                    idx = filtered_df[filtered_df.loc[:, "handness"] == 2.0].index
                    df.loc[idx, handedness] = 2.0
                    df.loc[idx, hgs_dominant] = df.loc[idx, f"46-{ses}.0"]
                    df.loc[idx, hgs_dominant_side] = "left"
                    df.loc[idx, hgs_nondominant] = df.loc[idx, f"47-{ses}.0"]
                    df.loc[idx, hgs_nondominant_side] = "right"

                    idx = filtered_df[filtered_df.loc[:, "handness"].isin([3.0, -3.0, np.NaN])].index
                    df.loc[idx, handedness] = 3.0
                    df_tmp = df.loc[idx, :]
                    if (df_tmp.loc[:, f"47-{ses}.0"] == df_tmp.loc[:, f"46-{ses}.0"]).all():
                        idx_tmp = df_tmp[df_tmp.loc[:, f"47-{ses}.0"] == df_tmp.loc[:, f"46-{ses}.0"]].index
                        df.loc[idx_tmp, hgs_dominant] = df.loc[idx, f"47-{ses}.0"]
                        df.loc[idx_tmp, hgs_dominant_side] = "balanced_hgs"
                        df.loc[idx_tmp, hgs_nondominant] = df.loc[idx, f"46-{ses}.0"]
                        df.loc[idx_tmp, hgs_nondominant_side] = "balanced_hgs"
                    elif (df_tmp.loc[:, f"47-{ses}.0"] != df_tmp.loc[:, f"46-{ses}.0"]).all():
                        idx_tmp = df_tmp[df_tmp.loc[:, f"47-{ses}.0"] != df_tmp.loc[:, f"46-{ses}.0"]].index
                        # Find the column with the maximum value among '46-{ses}.0' and '47-{ses}.0' for filtered rows
                        result_column = df.loc[idx_tmp, [f"46-{ses}.0", f"47-{ses}.0"]].idxmax(axis=1)
                        condition_right = result_column[result_column == f"47-{ses}.0"]
                        df.loc[condition_right.index, hgs_dominant] = df.loc[condition_right.index, f"47-{ses}.0"]
                        df.loc[condition_right.index, hgs_dominant_side] = "right"
                        df.loc[condition_right.index, hgs_nondominant] = df.loc[condition_right.index, f"46-{ses}.0"]
                        df.loc[condition_right.index, hgs_nondominant_side] = "left"
                        condition_left = result_column[result_column == f"46-{ses}.0"]
                        df.loc[condition_left.index, hgs_dominant] = df.loc[condition_left.index, f"46-{ses}.0"]
                        df.loc[condition_left.index, hgs_dominant_side] = "left"
                        df.loc[condition_left.index, hgs_nondominant] = df.loc[condition_left.index, f"47-{ses}.0"]
                        df.loc[condition_left.index, hgs_nondominant_side] = "right"
                                
        return df

################################ EXTRACT DATA ##############################
# The main goal of data validation is to verify that the data is 
# accurate, reliable, and suitable for the intended analysis.
###############################################################################
    def extract_data(self, df, session_column):
            """Exclude all subjects who had Dominant HGS < 4 and != NaN:

            Parameters
            ----------
            df : dataframe
                The dataframe that desired to analysis

            Return
            ----------
            df : dataframe
            """
            assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
            # -----------------------------------------------------------
            if df[session_column].isna().sum() < len(df):
                df = df[df[session_column]>=0]
                
            elif df[session_column].isna().sum() == len(df):
                # Drop all rows from the DataFrame
                df = pd.DataFrame(columns=df.columns)
                # Display message to the user
                print(f"******* No patient has assessed for {session_column} *******")
                
            return df

################################ DATA VALIDATION ##############################
# The main goal of data validation is to verify that the data is 
# accurate, reliable, and suitable for the intended analysis.
###############################################################################
    def validate_handgrips(self, df, session_column):
        """Exclude all subjects who had Dominant HGS < 4 and != NaN:

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
        """                
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        if df[session_column].isna().sum() < len(df):
            # Calculate Dominant HGS by
            # Calling the modules
            df = self.remove_missing_hgs_pre_post_disorder(df, session_column)
            hgs_dominant = session_column.replace(substring_to_remove, "hgs_dominant")
            hgs_nondominant = session_column.replace(substring_to_remove, "hgs_nondominant")
            # ------------------------------------
            # Exclude all subjects who had Dominant HGS < 4:
            # The condition is applied to "hgs_dominant" columns
            # And then reset_index the new dataframe:
            df = df[(~df.loc[:, hgs_dominant].isna()) & (~df.loc[:, hgs_nondominant].isna())]

        elif df.loc[:, session_column].isna().sum() == len(df):
            # Drop all rows from the DataFrame
            df = pd.DataFrame(columns=df.columns)
            # Display message to the user
            print(f"******* No patient has assessed for {session_column} *******")

        return df

###############################################################################
    def remove_missing_hgs_pre_post_disorder(self, df, session_column):
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        # -----------------------------------------------------------
         # Handgrip strength info
        # for Left and Right Hands
        hgs_left = "46"  # Handgrip_strength_(left)
        hgs_right = "47"  # Handgrip_strength_(right)
        
        for idx in df.index:
            session = df.loc[idx, session_column]
            # hgs_left field-ID: 46
            # hgs_right field-ID: 47
            if ((np.isnan(df.loc[idx, f"{hgs_left}-{session}"])) | (np.isnan(df.loc[idx, f"{hgs_right}-{session}"]))):
                    df = df.drop(index=idx)

        return df

###############################################################################
    def extract_post_disorder_df(self, df, mri_status):
        """Extract the post disorder dataframe from the disorder dataframe. 
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disorder.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        # UKB contains 4 assessment sessions
        if mri_status == "mri":
            sessions = 4
        elif mri_status == "nonmri":
            sessions = 2
        # Initialize an empty list
        followupdays_cols = []
        # Append "followup_days" column names to list
        for ses in range(0, sessions):
            followupdays_cols.append(f"followup_days-{ses}.0")

        # Create a boolean mask for rows where all numeric variables are greater than zero or are NaN
        mask = df[followupdays_cols].apply(lambda row: np.all(np.logical_or(np.isnan(row), row >= 0)), axis=1)

        # Apply the mask to the DataFrame to get the desired rows
        # for only Post-disorder subjects
        post_disorder_df = df[mask]
        
        return post_disorder_df

###############################################################################
    def extract_pre_disorder_df(self, df, mri_status):
        """Extract the pre disorder dataframe from the disorder dataframe. 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disorder.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        if mri_status == "mri":
            sessions = 4
        elif mri_status == "nonmri":
            sessions = 2
        followupdays_cols = []
        for ses in range(0, sessions):
            followupdays_cols.append(f"followup_days-{ses}.0")

        # Create a boolean mask for rows where all numeric variables are lower than zero or are NaN
        mask = df[followupdays_cols].apply(lambda row: np.all(np.logical_or(np.isnan(row), row < 0)), axis=1)

        # Apply the mask to the DataFrame to get the desired rows
        # for only Pre-disorder subjects
        pre_disorder_df = df[mask]
        
        return pre_disorder_df

###############################################################################
    def extract_longitudinal_disorder_df(self, df, mri_status):
        """Extract the longitudinal dataframe from the disorder dataframe. 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disorder.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        df_pre_disorder = self.extract_pre_disorder_df(df, mri_status)
        df_post_disorder = self.extract_post_disorder_df(df, mri_status)

        # The intersection between pre and post dataframes of disorder
        # will be the longitudinal dataframe.
        # Find duplicate index values
        merged_df = pd.concat([df_pre_disorder, df_post_disorder])
        # Find duplicate index values on the original DataFrame
        longitudinal_disorder_df = df[~df.index.isin(merged_df.index)]
        
        return longitudinal_disorder_df

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
