#!/usr/bin/env Disorderspredwp3
"""Perform different preprocessing on disease."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import numpy as np
import pandas as pd
import math
from datetime import datetime as dt

from ptpython.repl import embed

###############################################################################
class PrepareDisease:
    def __init__(self, df, population):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe that desired to analysis.
        """
        self.df = df
        self.population = population

###############################################################################
    def remove_missing_disease_dates(self, df):
        """ Drop all subjects who has no date of disease and
            all dates of 1900-01-01 epresents "Date is unknown".
            
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disease.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.   
        """
        # Date of the earliest reported stroke for a participant.
        # The source of this report is given in Field 42007.
        # Coding 272:
        #           1900-01-01 represents "Date is unknown"

        population = self.population
        
        if population == "stroke":
            df = df[(~df['42006-0.0'].isna()) &
                                  (df['42006-0.0'] != "1900-01-01")]
        elif population == "parkinson":
            df = df[(~df['42030-0.0'].isna()) &
                                  (df['42030-0.0'] != "1900-01-01")]
            
        return df
    
###############################################################################
    def remove_missing_hgs(self, df):
        """ Check availability of Handgrip_strength on sessions level.
        Create a list of different sessions dataframes for HGS.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.

        Returns
        --------
        df: pandas.DataFrame
            DataFrame of data specified.

        """
        # Handgrip strength info
        # for Left and Right Hands
        hgs_left = "46"  # Handgrip_strength_(left)
        hgs_right = "47"  # Handgrip_strength_(right)
        # Sessions/ Instances info
        sessions = 4

        # Create an empty df for output
        df_output = pd.DataFrame()
        # Check non-Zero and non_NaN Handgrip strength
        # for Left and Right Hands
        for ses in range(0, sessions):
            df_tmp = df[
                ((~df[f'{hgs_left}-{ses}.0'].isna()) &
                (df[f'{hgs_left}-{ses}.0'] !=  0))
                & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
                (df[f'{hgs_right}-{ses}.0'] !=  0))
            ]
            df_output = pd.concat([df_output, df_tmp])

        # Drop the duplicated subjects
        # based on 'eid' column (subject ID)
        df_output = df_output[~df_output.index.duplicated(keep='first')]

        return df_output

###############################################################################
    def date_difference(self, attendance, onset):
        """Find the number of days between two given dates.
           Here for diseae onset and attendance at clinic.
            
        Parameters
        ----------
        attendance :  array
            The column of the baseline visit date when the subject visited clinic.
        onset : array
            The column of the disease onset date when the disease occurred.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        onset_date = pd.to_datetime(onset)
        attendance_date = pd.to_datetime(attendance)
        
        days = (attendance_date-onset_date).dt.days
        
        return days
    
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
        population = self.population

        sessions = 4
        
        if population == "stroke":
            onset = df['42006-0.0']
        elif population == "parkinson":
            onset = df['42030-0.0']
            
        for ses in range(0, sessions):
            attendance = df[f'53-{ses}.0']
            df[f'followup_days-{ses}.0'] = self.date_difference(attendance, onset)

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
        df['earliest_subtype'] = df_subtype.min(axis=1)
        df['stroke_subtype_field'] = df_subtype.idxmin(axis=1)
        
        df.loc[df[df['stroke_subtype_field']=="42008-0.0"].index, 'stroke_subtype'] = "ischaemic"
        df.loc[df[df['stroke_subtype_field']=="420010-0.0"].index, 'stroke_subtype'] = "intracerebral haemorrhage"
        df.loc[df[df['stroke_subtype_field']=="420012-0.0"].index, 'stroke_subtype'] = "subarachnoid haemorrhage"

        return df

###############################################################################
    def define_pre_post_longitudinal(self, df):
        
        
        filter_col = [col for col in df if col.startswith('followup_days')]
        post_df = df[df[filter_col].min(axis=1)>=0]
        df.loc[post_df.index,"stroke_cohort"] = "post-stroke"
        df.loc[post_df.index,"num_pre_sessions"] = 0
        df.loc[post_df.index,"num_post_sessions"] =  post_df[post_df[filter_col]>=0].count(axis=1)
              
        
        pre_df = df[df[filter_col].max(axis=1) < 0]
        df.loc[pre_df.index,"stroke_cohort"] = "pre-stroke"
        df.loc[pre_df.index,"num_pre_sessions"] = pre_df[pre_df[filter_col]<0].count(axis=1)
        df.loc[pre_df.index,"num_post_sessions"] = 0

        # To union the two post and pre DataFrames together:
        pre_post_df = pd.concat([pre_df, post_df])
        # The intersection between pre and post dataframes of disease
        # will be the longitudinal dataframe.
        longitudinal_df = df[~df.index.isin(pre_post_df.index)]
        df.loc[longitudinal_df.index,"num_pre_sessions"] = longitudinal_df[longitudinal_df[filter_col]<0].count(axis=1)
        df.loc[longitudinal_df.index,"num_post_sessions"] = longitudinal_df[longitudinal_df[filter_col]>=0].count(axis=1)
        df.loc[longitudinal_df.index,"stroke_cohort"] = "longitudinal_stroke"
        
        
        return df
    
###############################################################################
    def define_recovery_data(self, df):
        
        filter_col = [col for col in df if col.startswith('followup_days')]

        # Identify positive values
        positive_values = df[filter_col] > 0

        # Count positive values in each row
        positive_counts = positive_values.sum(axis=1)

        # Select rows with at least two positive values
        df = df[positive_counts >= 2]
        
        return df
###############################################################################
    def define_all_post_subjects(self, df):
            
            filter_col = [col for col in df if col.startswith('followup_days')]

            # Identify positive values
            positive_values = df[filter_col] > 0

            # Count positive values in each row
            positive_counts = positive_values.sum(axis=1)

            # Select rows with at least two positive values
            df = df[positive_counts >= 1]
            
            return df
###############################################################################
    def define_hgs(self, df):

        left_hgs = "46"
        right_hgs = "47"

        filter_col = [col for col in df if col.endswith('pre_session')]
        for j in range(0,4):
            ses = df[filter_col].iloc[:,j].astype(str).str[8:]
            for i in range(0, len(df[filter_col])):
                idx=ses.index[i]
                if ses.iloc[i] != "":
                    df.loc[idx, f"{j+1}_pre_left_hgs"] = df.loc[idx, f"{left_hgs}-{ses.iloc[i]}"]
                    df.loc[idx, f"{j+1}_pre_right_hgs"] = df.loc[idx, f"{right_hgs}-{ses.iloc[i]}"]
                else:
                    df.loc[idx, f"{j+1}_pre_left_hgs"] = np.NaN
                    df.loc[idx, f"{j+1}_pre_right_hgs"] = np.NaN
                    
        filter_col = [col for col in df if col.endswith('post_session')]
        for j in range(0,4):
            ses = df[filter_col].iloc[:,j].astype(str).str[8:]
            for i in range(0, len(df[filter_col])):
                idx=ses.index[i]
                if ses.iloc[i] != "":
                    df.loc[idx, f"{j+1}_post_left_hgs"] = df.loc[idx, f"{left_hgs}-{ses.iloc[i]}"]
                    df.loc[idx, f"{j+1}_post_right_hgs"] = df.loc[idx, f"{right_hgs}-{ses.iloc[i]}"]
                else:
                    df.loc[idx, f"{j+1}_post_left_hgs"] = np.NaN
                    df.loc[idx, f"{j+1}_post_right_hgs"] = np.NaN

        return df

###############################################################################
    def check_hgs(self, df):

        df_output = pd.DataFrame()
        
        for j in range(0,4):
            df_tmp_pre = df[((~df[f"{j+1}_pre_left_hgs"].isna()) & (df[f"{j+1}_pre_left_hgs"] !=  0)) 
                        & ((~df[f"{j+1}_pre_right_hgs"].isna()) & (df[f"{j+1}_pre_right_hgs"] !=  0))
                        ]
            df_tmp_post = df[((~df[f"{j+1}_post_left_hgs"].isna()) & (df[f"{j+1}_post_left_hgs"] !=  0)) 
                        & ((~df[f"{j+1}_post_right_hgs"].isna()) & (df[f"{j+1}_post_right_hgs"] !=  0))
                        ]
            df_output = pd.concat([df_output, df_tmp_pre, df_tmp_post])

        df_output = df_output[~df_output.index.duplicated(keep='first')]

        return df_output
    
###############################################################################   
    # Function to get column names with first to fourth sorted values
    def define_pre_post_sessions(self, df):
        filter_col = [col for col in df if col.startswith('followup_days')]
        for i in range(0, len(df[filter_col])):
            sorted_values = df[filter_col].iloc[i].sort_values()
            positive_numbers= sorted_values[sorted_values>0]
            positive_rest = 4 - len(positive_numbers)
            sorted_values_neg = df[filter_col].iloc[i].sort_values(ascending=False)
            negative_numbers = sorted_values_neg[sorted_values_neg<0]
            negative_rest = 4 - len(negative_numbers)
            # Create new columns based on the length of the series
            df_tmp_positive = pd.DataFrame(columns=[f"{j+1}_post_session" for j in range(len(positive_numbers))])
            df_tmp_rest_positive = pd.DataFrame(columns=[f"{len(positive_numbers)+j+1}_post_session" for j in range(positive_rest)])
            # Assign the values from the series to the new columns
            if len(positive_numbers) > 0:
                df_tmp_positive.loc[sorted_values.name] = positive_numbers.index
                df_tmp_rest_positive.loc[sorted_values.name] = np.NaN
                df_positive = pd.concat([df_tmp_positive, df_tmp_rest_positive], axis=1)
            elif len(positive_numbers) == 0:
                df_tmp_rest_positive.loc[sorted_values.name] = np.NaN
                df_positive = df_tmp_rest_positive
            
            # Create new columns based on the length of the series
            df_tmp_negative = pd.DataFrame(columns=[f"{j+1}_pre_session" for j in range(len(negative_numbers))])
            df_tmp_rest_negative = pd.DataFrame(columns=[f"{len(negative_numbers)+j+1}_pre_session" for j in range(negative_rest)])
            # Assign the values from the series to the new columns
            # print("===== Done! =====")
            # embed(globals(), locals())
            if len(negative_numbers) > 0:
                df_tmp_negative.loc[sorted_values_neg.name] = negative_numbers.index
                df_tmp_rest_negative.loc[sorted_values_neg.name] = np.NaN
                df_negative = pd.concat([df_tmp_negative, df_tmp_rest_negative], axis=1)

            elif len(negative_numbers) == 0:
                df_tmp_rest_negative.loc[sorted_values_neg.name] = np.NaN
                df_negative = df_tmp_rest_negative
                
            df_tmp = pd.concat([df_negative, df_positive], axis=1)
            # print("===== Done! =====")
            # embed(globals(), locals())
            
            for col in range(df_tmp.shape[1]):
                if any(str(item).lower() == 'nan' for item in df_tmp.iloc[:, col].values):
                    df.loc[df.index==df_tmp.index[0], df_tmp.columns[col]] = df_tmp.iloc[:, col]
                    # df.loc[df.index==sorted_values.name, df_tmp.columns[col]] = df_tmp.iloc[:, col]

                else:
                    # df.loc[df.index==sorted_values.name, df_tmp.columns[col]] = f"session-{df_tmp.iloc[:, col].values[0][14:]}"
                    df.loc[df.index==df_tmp.index[0], df_tmp.columns[col]] = f"session-{df_tmp.iloc[:, col].values[0][14:]}"
            
        return df
    
###############################################################################
    def extract_post_disease(self, df):
        """Extract the post disease dataframe from the disease dataframe. 
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disease.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        post_df = df[df['stroke_cohort']=='post-stroke']    

        return post_df

###############################################################################
    def extract_pre_disease(self, df):
        """Extract the pre disease dataframe from the disease dataframe. 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
        population : str
            Name of population/disease.
            
        Returns
        --------
        df : pandas.DataFrame
            DataFrame of data specified.
        """
        
        # filter_col = [col for col in df if col.startswith('followup_days')]
        # index = np.where(df[filter_col].max(axis=1) < 0)[0]
        # df = df.loc[index, :]
        
        pre_df = df[df['stroke_cohort']=='pre-stroke']
        
        return pre_df

###############################################################################
    def extract_longitudinal_disease(self, df):
        """Extract the longitudinal dataframe from the disease dataframe. 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of data specified.
            
        Returns
        --------
        longitudinal_df : pandas.DataFrame
            DataFrame of data specified.
        """

        # post_df = self.extract_post_disease(df)
        # pre_df = self.extract_pre_disease(df)
        # # To union the two post and pre DataFrames together:
        # pre_post_df = pd.concat([pre_df, post_df])
        
        # # The intersection between pre and post dataframes of disease
        # # will be the longitudinal dataframe.
        # index = df.index.symmetric_difference(pre_post_df.index)
        # longitudinal_df = df.loc[index, :]
        
        longitudinal_df = df[df['stroke_cohort']=='longitudinal_stroke']
        
        return longitudinal_df

###############################################################################
    def disease_subsets(self, dataframe, population):
        
        sessions = 4
        days = [col for col in dataframe.columns if 'days_to_onset' in col]
        df_post = dataframe[dataframe[days]>=0]

        first_post_visit_days = df_post[days].min(axis=1)
        first_post_visit_ses = df_post[days].idxmin(axis=1)

        df_pre = dataframe[dataframe[days]<0]

        first_pre_visit_days = df_pre[days].max(axis=1)
        first_pre_visit_ses = df_pre[days].idxmax(axis=1)

        dataframe['first_post_visit_days'] = first_post_visit_days
        dataframe['first_post_visit_ses'] = first_post_visit_ses
        dataframe['first_post_visit_ses'] = dataframe['first_post_ses'].replace(to_replace=days, value=[0, 1, 2, 3])

        for ses in range(0, sessions):
            post_ses_subs = dataframe.loc[dataframe['first_post_ses'] == ses, 'eid']
            dataframe.loc[dataframe['first_post_ses'] == ses, 'first_post_hgs_L'] = dataframe[f'46-{ses}.0']
            dataframe.loc[dataframe['first_post_ses'] == ses, 'first_post_hgs_R'] = dataframe[f'47-{ses}.0']

        for ses in range(0, sessions):
            dataframe.loc[dataframe['first_post_ses'] == ses, 'first_post_BMI'] = dataframe[f'21002-{ses}.0']


        dataframe['first_pre_days'] = first_pre_visit_days
        dataframe['first_pre_ses'] = first_pre_visit_ses
        dataframe['first_pre_ses'] = dataframe['first_pre_ses'].replace(to_replace=days, value=[0, 1, 2, 3])
        for ses in range(0, sessions):
            pre_ses_subs = dataframe.loc[dataframe['first_pre_ses'] == ses, 'eid']
            dataframe.loc[dataframe['first_pre_ses'] == ses, 'first_pre_hgs_L'] = dataframe[f'46-{ses}.0']
            dataframe.loc[dataframe['first_pre_ses'] == ses, 'first_pre_hgs_R'] = dataframe[f'47-{ses}.0']

        for ses in range(0, sessions):
            dataframe.loc[dataframe['first_pre_ses'] == ses, 'first_pre_BMI'] = dataframe[f'21002-{ses}.0']

        return dataframe

###############################################################################
    def howmany_sessions(
        self,
        dataframe,
        num_session,
    ):
        total_days_col = [col for col in dataframe.columns if 'total' in col]
        sub_df_days = pd.DataFrame(dataframe, columns=total_days_col)

        dataframe['num_NaN_sessions'] = sub_df_days.isna().sum(axis=1)

        return dataframe

###############################################################################
    def prior_stroke_subs(
        self,
        dataframe,
        num_session,
    ):
        dataframe = self.cal_stroke_based_days(dataframe, num_session)
        total_days_col = [col for col in dataframe.columns if 'total' in col]
        sub_df_days = pd.DataFrame(dataframe, columns=total_days_col)

        sub_df_days[total_days_col] = sub_df_days[total_days_col].apply(
                lambda x: pd.to_numeric(x, errors='coerce'))
        # dataframe = dataframe [(dataframe[total_days_col].isna().sum(axis=1) == 3)]

        first_prior_strok_days = sub_df_days.where(sub_df_days<0).max(axis = 1)
        first_prior_strok_ses = sub_df_days.where(sub_df_days<0).idxmax(axis = 1)
        
        dataframe['first_prior_days'] = first_prior_strok_days
        dataframe['first_prior_session'] = first_prior_strok_ses
        dataframe['first_prior_session'] = dataframe['first_prior_session'].replace(to_replace=total_days_col, value=[0, 1, 2, 3])

        for ses in range(0, num_session):
            prior_ses_subs = dataframe.loc[dataframe['first_prior_session'] == ses, 'eid']
            dataframe.loc[dataframe['first_prior_session'] == ses, 'first_prior_hgs_L'] = dataframe[f'46-{ses}.0']
            dataframe.loc[dataframe['first_prior_session'] == ses, 'first_prior_hgs_R'] = dataframe[f'47-{ses}.0']
        
        dataframe = dataframe.dropna(subset = ['first_prior_hgs_L', 'first_prior_hgs_R'])

        return dataframe

###############################################################################
    def first_visit_poststroke(
        self,
        dataframe,
        num_session,
    ):
        dataframe = self.cal_stroke_based_days(dataframe, num_session)
        total_days_col = [col for col in dataframe.columns if 'total' in col]
        sub_df_days = pd.DataFrame(dataframe, columns=total_days_col)
        sub_df_days[total_days_col] = sub_df_days[total_days_col].apply(
                lambda x: pd.to_numeric(x, errors='coerce'))
        # first_post_strok_days = dataframe.where(dataframe[total_days_col] >=0).min(axis=1)
        # dataframe = dataframe[dataframe[total_days_col] >=0]
        # dataframe = dataframe[dataframe[total_days_col].isna().sum(axis=1) == 3]
        # dataframe = dataframe [(dataframe[total_days_col].isna().sum(axis=1) == 3)]
        # # a = sub_df_days.where(sub_df_days[total_days_col].isna().sum(axis=1) == 3)
        # first_post_strok_days = dataframe.where(dataframe[total_days_col]>=0).min(axis = 1)
        # first_post_strok_ses = dataframe.where(dataframe[total_days_col]>=0).idxmin(axis = 1)
        
        # dataframe = dataframe.where(sub_df_days[total_days_col].isna().sum(axis=1) == 3)
        df_post = sub_df_days[sub_df_days['total_days_ses-0.0']>=0]
        first_post_strok_days = df_post[total_days_col].min(axis = 1)
        first_post_strok_ses = dataframe[total_days_col].idxmin(axis = 1)
        sub_df_days['first_visit_days'] = first_post_strok_days
        sub_df_days['first_visit_session'] = first_post_strok_ses
        sub_df_days['first_visit_session'] = sub_df_days['first_visit_session'].replace(to_replace=total_days_col, value=[0, 1, 2, 3])

        dataframe['first_visit_days'] = first_post_strok_days
        dataframe['first_visit_session'] = first_post_strok_ses
        dataframe['first_visit_session'] = dataframe['first_visit_session'].replace(to_replace=total_days_col, value=[0, 1, 2, 3])

        for ses in range(0, num_session):
            min_ses_subs = dataframe.loc[dataframe['first_visit_session'] == ses, 'eid']
            dataframe.loc[dataframe['first_visit_session'] == ses, 'first_visit_hgs_L'] = dataframe[f'46-{ses}.0']
            dataframe.loc[dataframe['first_visit_session'] == ses, 'first_visit_hgs_R'] = dataframe[f'47-{ses}.0']
        
        dataframe = dataframe.dropna(subset = ['first_visit_hgs_L', 'first_visit_hgs_R'])

        dataframe = dataframe[dataframe['total_days_ses-0.0']<0]
        first_prior_strok_days = dataframe[total_days_col].max(axis = 1)
        first_prior_strok_ses = dataframe[total_days_col].idxmax(axis = 1)
        
        dataframe['first_prior_days'] = first_prior_strok_days
        dataframe['first_prior_session'] = first_prior_strok_ses
        dataframe['first_prior_session'] = dataframe['first_prior_session'].replace(to_replace=total_days_col, value=[0, 1, 2, 3])

        for ses in range(0, num_session):
            prior_ses_subs = dataframe.loc[dataframe['first_prior_session'] == ses, 'eid']
            dataframe.loc[dataframe['first_prior_session'] == ses, 'first_prior_hgs_L'] = dataframe[f'46-{ses}.0']
            dataframe.loc[dataframe['first_prior_session'] == ses, 'first_prior_hgs_R'] = dataframe[f'47-{ses}.0']
        
        dataframe = dataframe.dropna(subset = ['first_prior_hgs_L', 'first_prior_hgs_R'])

        return dataframe

# -------------------------------
    def add_mri_status(
        self,
        non_mri_dataframe,
        mri_dataframe,
    ):

        non_mri_dataframe['MRI'] = non_mri_dataframe['eid'].isin(mri_dataframe['eid'])
        non_mri_dataframe['MRI'].replace({True: 1, False: 0}, inplace = True)

        return non_mri_dataframe
###############################################################################
    def check_hgs_for_stroke(
        self,
        dataframe,
    ):
        dataframe['HGS_status'] = dataframe.mask(~dataframe['46-0.0'].isna(), 1)
        dataframe['HGS_status'] = dataframe.mask(dataframe['46-0.0'].isna(), 0)

        # dataframe['HGS_status'].replace({True: 1, False: 0}, inplace = True)
###############################################################################
    def subtraction_hgs(
        self,
        dataframe,
        HGS_L,
        HGs_R
    ):
        dataframe['sub_hgs'] = dataframe['HGS_L'] - dataframe['HGS_R']

        return dataframe
###############################################################################
    def sum_hgs(
        self,
        dataframe,
    ):
        dataframe['sum_hgs'] = dataframe['first_visit_hgs_L'] + dataframe['first_visit_hgs_R']

        return dataframe
###############################################################################
    def laterality_index(
        self,
        dataframe,
        HGS_L,
        HGS_R
    ):
        dataframe['LI'] =  (dataframe[HGS_L]-dataframe[HGS_R]) / (dataframe[HGS_L]+dataframe[HGS_R])
        dataframe['abs_LI'] = abs(dataframe[HGS_L]-dataframe[HGS_R]) / abs(dataframe[HGS_L]+dataframe[HGS_R])

        return dataframe
###############################################################################
    def remove_stroke_before_2006(
        self,
        dataframe,
        stroke_date
    ):
        start_date = pd.to_datetime('2005-01-01')
        stroke_date = pd.to_datetime(stroke_date)
        dataframe = dataframe[stroke_date > start_date]

        return dataframe

###############################################################################
    def post_stroke_6months(
        self,
        dataframe,
    ):
        start_date = pd.to_datetime('2005-01-01')
        stroke_date = pd.to_datetime(stroke_date)
        dataframe = dataframe[stroke_date > start_date]

        return dataframe

###############################################################################
    def add_mri_status(
        self,
        non_mri_dataframe,
        mri_dataframe,
    ):
        non_mri_dataframe['MRI'] = non_mri_dataframe['eid'].isin(mri_dataframe['eid'])
        non_mri_dataframe['MRI'].replace({True: 1, False: 0}, inplace = True)
        
        return non_mri_dataframe
# -----------------------------------------------------------------------------#    

