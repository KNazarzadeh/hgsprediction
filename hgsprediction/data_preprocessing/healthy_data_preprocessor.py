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
class DataPreprocessor:
    def __init__(
        self,
        df: pd.DataFrame,
        session,
    ):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df
        self.session = session
        
################################ DATA VALIDATION ##############################
# The main goal of data validation is to verify that the data is 
# accurate, reliable, and suitable for the intended analysis.
###############################################################################

    def validate_handgrips(
        self,
        df,
    ):
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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # ------------------------------------
        # Calculate Dominant and Non-Dominant HGS by
        # Calling the modules:
        df = self.dominant_handgrip(df)
        df = self.nondominant_handgrip(df)
        # ------------------------------------
        # Exclude all subjects who had Dominant HGS < 4:
        # The condition is applied to "dominant_hgs" columns
        # And then reset_index the new dataframe:
        df = df[df.loc[:, f"dominant_hgs-{session}.0"] >=4]

        return df

###############################################################################
    def dominant_handgrip(
        self,
        df,
    ):
        """Calculate dominant handgrip
        and add "dominant_hgs" column to dataframe

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
        # index = np.where(df.loc[:, f"1707-{session}.0"] == 1.0)[0]
        index = df[df.loc[:,f"1707-{session}.0"] == 1.0].index
        # Add and new column "dominant_hgs"
        # And assign Right hand HGS value:
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, f"47-{session}.0"]
        # ------------------------------------
        # If handedness is equal to 2
        # Left hand is Dominant
        # Find handedness equal to 2:
        # index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        index = df[df.loc[:,f"1707-{session}.0"] == 2.0].index
        # Add and new column "dominant_hgs"
        # And assign Left hand HGS value:
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, f"46-{session}.0"]
        # ------------------------------------
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer) OR
        # NaN value
        # Dominant will be the Highest Handgrip score from both hands.
        # Find handedness equal to 3, -3 or NaN:
        # index = np.where(
        #     (df.loc[:, f"1707-{session}.0"] == 3.0) |
        #     (df.loc[:, f"1707-{session}.0"] == -3.0)|
        #     (df.loc[:, f"1707-{session}.0"].isna()))[0]
        index = df[(df.loc[:, f"1707-{session}.0"] == 3.0) |
                   (df.loc[:, f"1707-{session}.0"] == -3.0)|
                   (df.loc[:, f"1707-{session}.0"].isna())].index
        # Add and new column "dominant_hgs"
        # And assign Highest HGS value among Right and Left HGS:        
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, [f"46-{session}.0", f"47-{session}.0"]].max(axis=1)
            
        return df

###############################################################################
    def nondominant_handgrip(
        self,
        df,
    ):
        """Calculate dominant handgrip
        and add "nondominant_hgs" column to dataframe
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
        # Add and new column "nondominant_hgs"
        # And assign Left hand HGS value:
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[:, f"46-{session}.0"]
        # ------------------------------------
        # If handedness is equal to 2
        # Right hand is Non-Dominant
        # Find handedness equal to 2:
        # index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        index = df[df.loc[:,f"1707-{session}.0"] == 2.0].index
        # Add and new column "nondominant_hgs"
        # And assign Right hand HGS value:        
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
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
        index = df[(df.loc[:, f"1707-{session}.0"] == 3.0) |
                   (df.loc[:, f"1707-{session}.0"] == -3.0)|
                   (df.loc[:, f"1707-{session}.0"].isna())].index
        # Add and new column "nondominant_hgs"
        # And assign Lowest HGS value among Right and Left HGS:    
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[:, [f"46-{session}.0", f"47-{session}.0"]].min(axis=1)
    
        return df

############################## FFEATURE ENGINEERING ###########################
# Creating new columns/features/targets from existing data
# Preprocess features or Handling Outliers
# more meaningful insights and patterns for machine learning models.
###############################################################################
    def calculate_waist_to_hip_ratio(
        self,
        df,
    ):
        """Calculate and add "Waist to Hip Ratio" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: Waist to Hip Ratio
        """
        # Assign corresponding session number from the Class:
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # ------------------------------------
        # Add new column "waist_to_hip_ratio" by the following process: 
        # Waist circumference field-ID: 48
        # Hip circumference field-ID: 49
        # Calculating Waist/Hip
        df.loc[:, f"waist_to_hip_ratio-{session}.0"] = \
            (df.loc[:, f"48-{session}.0"].astype(str).astype(float)).div(
                df.loc[:, f"49-{session}.0"].astype(str).astype(float))

        return df

###############################################################################
    def sum_handgrips(
        self,
        df,
    ):
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
    def calculate_left_hgs(
        self,
        df,
    ):
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
    def calculate_right_hgs(
        self,
        df,
    ):
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
    def sub_handgrips(
        self,
        df,
    ):
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
    def calculate_laterality_index(
        self,
        df,
    ):
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

        df_sub = self.sub_handgrips(df)
        df_sum = self.sum_handgrips(df)

        # Laterality Index of Handgrip strength: (Left - Right)/(Left +right)
        df.loc[:, f"hgs(LI)-{session}.0"] = \
            (df_sub.loc[:, f"hgs(L-R)-{session}.0"] / df_sum.loc[:, f"hgs(L+R)-{session}.0"])

        return df

###############################################################################
    def calculate_neuroticism_score(
        self,
        df,
    ):
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
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
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
        # Add corresponding intences/session that we are looking for 
        # to the list of fields as suffix:
        neuroticism_fields = [s + f"-{session}.0" for s in neuroticism_fields]
        # ------------------------------------
        # Add new column "neuroticism_score" by the following process: 
        # Find core_fields answered greather than 0.0
        # And Calculate Sum with min_count parameter=9:
        # df.where is replacing all negative values with NaN
        # min_count=9, means calculate Sum if 
        # at leaset 9 of the fields are answered:
        # df.loc[:, "neuroticism_score"] = \
        #     df.where(df.loc[:, neuroticism_fields] >= 0.0).sum(axis=1, min_count=9)
        df.loc[:, "neuroticism_score"] = df[neuroticism_fields].where(df.loc[:, neuroticism_fields] >= 0.0).sum(axis=1, min_count=9)

        return df

###############################################################################
    def calculate_depression_score(
        self,
        df,
    ):
        """Calculate depression score
        and add "depression_score" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: depression score
        """
        # Assign corresponding session number from the Class:
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # -----------------------------------------------------------
        # ------- Depression Fields and Data-Codings -------
        # All Fields useing same Data-Coding (504)
        # Data-Coding: 504
        #          -818	Prefer not to answer
        #           1	Not at all
        #           2	Several days
        #           3	More than half the days
        #           4	Nearly every day
        # ------------------------------------
        #  ------- Depression Fields:
            # Recent feelings of inadequacy
            # '20507',  Data-Coding 504
            # trouble concentrating on things
            # '20508',  Data-Coding 504
            # Recent feelings of depression
            # '20510',
            # Recent poor appetite or overeating
            # '20511',  Data-Coding 504
            # Recent thoughts of suicide or self-harm
            # '20513',  Data-Coding 504
            # Recent lack of interest or pleasure in doing things
            # '20514',  Data-Coding 504
            # Trouble falling or staying asleep, or sleeping too much
            # '20517',  Data-Coding 504
            # Recent changes in speed/amount of moving or speaking
            # '20518',  Data-Coding 504
            # Recent feelings of tiredness or low energy
            # '20519',  Data-Coding 504
        # -----------------------------------------------------------
        depression_fields = [
            '20507',    # Data-Coding 504
            '20508',    # Data-Coding 504
            '20510',    # Data-Coding 504
            '20511',    # Data-Coding 504
            '20513',    # Data-Coding 504
            '20514',    # Data-Coding 504
            '20517',    # Data-Coding 504
            '20518',    # Data-Coding 504
            '20519',    # Data-Coding 504
        ]
        # Add corresponding intences/session that we are lookin for 
        # to the list of fields:
        depression_fields = [s + f"-{session}.0" for s in depression_fields]
        # ------------------------------------
        # Add new column "depression_score" by the following process: 
        # Find core_fields answered greather than 0.0
        # And Calculate Sum with min_count parameter=1:
        # df.where is replacing all negative values with NaN
        # min_count=1, means calculate Sum if 
        # at leaset 1 of the fields are answered:
        # df.loc[:, "depression_score"] = df.where(
        #     df[depression_fields] > 0.0).sum(axis=1, min_count=1)
        df.loc[:, "depression_score"] = df[depression_fields].where(df[depression_fields] > 0.0).sum(axis=1, min_count=1)

        return df

###############################################################################
    def calculate_anxiety_score(
        self,
        df,
    ):
        """Calculate anxiety score
        and add "anxiety_score" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: anxiety score
        """
        # Assign corresponding session number from the Class:
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # -----------------------------------------------------------
        # ------- Anxiety Fields and Data-Codings -------
        # All Fields useing same Data-Coding (504)
        # Data-Coding: 504
        #          -818	Prefer not to answer
        #           1	Not at all
        #           2	Several days
        #           3	More than half the days
        #           4	Nearly every day
        # ------------------------------------
        #  ------- Anxiety Fields:
        # Recent easy annoyance or irritability
        # '20505', Data-Coding 504
        # Recent feelings or nervousness or anxiety
        # '20506', Data-Coding 504
        # Recent inability to stop or control worrying
        # '20509', Data-Coding 504
        # Recent feelings of foreboding
        # '20512', Data-Coding 504
        # Recent trouble relaxing
        # '20515', Data-Coding 504
        # Recent restlessness
        # '20516', Data-Coding 504
        # Recent worrying too much about different things
        # '20520', Data-Coding: 504
        # -----------------------------------------------------------
        anxiety_fields = [
            '20505',    # Data-Coding: 504
            '20506',    # Data-Coding: 504
            '20509',    # Data-Coding: 504
            '20512',    # Data-Coding: 504
            '20515',    # Data-Coding: 504
            '20516',    # Data-Coding: 504
            '20520',    # Data-Coding: 504
        ]
        # Add corresponding intences/session that we are lookin for 
        # to the list of fields:
        anxiety_fields = [s + f"-{session}.0" for s in anxiety_fields]
        # ------------------------------------
        # Add new column "anxiety_score" by the following process: 
        # Find core_fields answered greather than 0.0
        # And Calculate Sum with min_count parameter=1:
        # df.where is replacing all negative values with NaN
        # min_count=1, means calculate Sum if 
        # at leaset 1 of the fields are answered:
        # df.loc[:, "anxiety_score"] = \
        #     df.where(df[anxiety_fields] > 0.0).sum(axis=1, min_count=1)
        df.loc[:, "anxiety_score"] = df[anxiety_fields].where(df[anxiety_fields] > 0.0).sum(axis=1, min_count=1)

        return df

###############################################################################
    def calculate_cidi_score(
        self,
        df,
    ):
        """Calculate CIDI score
        and add "cidi_score" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: CIDI score
        """
        # Assign corresponding session number from the Class:
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # -----------------------------------------------------------
        # ------- CIDI symptyms of depression -------
        # This field divided in 2 scores:
        # CIDI Core Fields and CIDI Non-Core Fields
        # ------------------------------------
        # ------- CIDI core symptyms of depression
        # Data-Coding: 503
        #           -818	Prefer not to answer
        #           0	    No
        #           1	    Yes
        # ------------------------------------
        # -------- CIDI core Fields --------
        # Ever had prolonged loss of interest in normal activities
        # '20441',    # Data-Coding: 503
        # # Ever had prolonged feelings of sadness or depression
        # '20446',    # Data-Coding: 503
        # ------------------------------------
        # ------- CIDI non-core Fields and Data-Codings
        # Data-Coding: 502
        #           -818	Prefer not to answer
        #           -121	Do not know
        #           0	    No
        #           1	    Yes
        # ------------------------------------
        # Data-Coding: 507
        #           -818	Prefer not to answer
        #           -121	Do not know
        #           0	    Stayed about the same or was on a diet
        #           1	    Gained weight
        #           2	    Lost weight
        #           3	    Both gained and lost some weight during the episode
        # -------- CIDI non-core Fields --------
        # Feelings of tiredness during worst episode of depression
        # '20449',    # Data-Coding 502
        # Feelings of worthlessness during worst period of depression
        # '20450',    # Data-Coding 502
        # Thoughts of death during worst depression
        # '20437',    # Data-Coding 502
        # Difficulty concentrating during worst depression
        # '20435',    # Data-Coding 502
        # Did your sleep change?
        # '20532',    # Data-Coding 502
        # Weight change during worst episode of depression
        # '20536',    # Data-Coding 507
        # -----------------------------------------------------------
        cidi_core_fields = [
            '20441',    # Data-Coding: 503
            '20446',    # Data-Coding: 503
        ]
        cidi_noncore_fields = [
            '20449',    # Data-Coding 502
            '20450',    # Data-Coding 502
            '20437',    # Data-Coding 502
            '20435',    # Data-Coding 502
            '20532',    # Data-Coding 502
            '20536',    # Data-Coding 507
        ]

        # Add corresponding intences/session that we are lookin for 
        # to the list of fields:
        cidi_core_fields = [s + f"-{session}.0" for s in cidi_core_fields]
        cidi_noncore_fields = [s +
                               f"-{session}.0" for s in cidi_noncore_fields]
        # ------------------------------------
        # Find Non-Core field '20536' if is in the following answers:
        #   1	Gained weight
        #   2	Lost weight
        #   3	Both gained and lost some weight during the episode
        # index = np.where(
        #     df.loc[:, f"20536-{session}.0"].isin([1.0, 2.0, 3.0]))[0]
        index = df[df.loc[:, f"20536-{session}.0"].isin([1.0, 2.0, 3.0])].index
        # And Replace with 1.0 value:
        df.loc[index, f"20536-{session}.0"] = 1.0
        # ------------------------------------
        # Find core_fields answered eaual or greather than 0.0
        # And Calculate Sum with min_count parameter=1:
        # df.where is replacing all negative values with NaN
        # min_count=1, means calculate Sum if 
        # at leaset 1 of the fields are answered:
        # core_score = df.where(
        #     df[cidi_core_fields] >= 0.0).sum(axis=1, min_count=1)
        core_score = df[cidi_core_fields].where(df[cidi_core_fields] >= 0.0).sum(axis=1, min_count=1)
        # Replace all core scores equal to 2.0 with 1.0 value:
        core_score = core_score.replace(2.0, 1.0)
        # ------------------------------------
        # Find Non-core_fields answered equal or greather than 0.0
        # And Calculate Sum with min_count parameter=4:
        # df.where is replacing all negative values with NaN
        # min_count=4, means calculate Sum if 
        # at leaset 4 of the fields are answered:
        # noncore_score = df.where(
        #     df[cidi_noncore_fields] >= 0.0).sum(axis=1, min_count=4)
        noncore_score = df[cidi_noncore_fields].where(df[cidi_noncore_fields] >= 0.0).sum(axis=1, min_count=4)
        # ------------------------------------
        # calculate cidi main score by adding core and non-core scores:
        # 'fill_value' parameter use when adding two series.
        # It means that any non NaN value be successful on adding.
        cidi_score = core_score.add(noncore_score, fill_value=0)
        # If cidi_score is nan or core_score are 'NaN'
        # then replace cidi_score with 'NaN':
        # Find cidi_score or core_score are 'NaN'
        index = cidi_score.isna() | core_score.isna()
        # And Replace cidi_score with 'NaN':
        cidi_score.loc[index] = np.NaN
        # ------------------------------------
        # Notice:
        # All Non-Core questions asked only when 
        # one of the Core was answered 'yes'.
        # So for those subjects that answered 
        # both Core questions as no (i.e. 0.0) we use cidi score 0.
        # ------------------------------------
        # Add new column "CIDI_score" by 
        # assigning cidi main score calculated above:
        df.loc[:, "CIDI_score"] = cidi_score

        return df

###############################################################################
    def calculate_qualification(
        self,
        df,
    ):
        """Calculate maximum qualification
        and add "qualification" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: maximum qualification
        """
        # Assign corresponding session number from the Class:
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        
        # ------- Qualification -------
        # '6138', Qualifications
        # Data-coding: 100305
        #           1	College or University degree
        #           2	A levels/AS levels or equivalent
        #           3	O levels/GCSEs or equivalent
        #           4	CSEs or equivalent
        #           5	NVQ or HND or HNC or equivalent
        #           6	Other professional qualifications eg: nursing, teaching
        #           -7	None of the above
        #           -3	Prefer not to answer
        # ------------------------------------
        # Array indices run from 0 to 5 for each intences:
        for i in range(6):
            # Find College or University degree
            # And Replace with '1.0' value
            index_college = np.where(
                df.loc[:,f"6138-{session}.{i}"] == 1.0)[0]
            df.loc[index_college, f"6138-{session}.{i}"] = 1.0
            # Find No College or University degree
            # And Replace with '0.0' value:
            index_no_college = np.where(
                df.loc[:,f"6138-{session}.{i}"].isin([2.0,
                                                      3.0,
                                                      4.0,
                                                      5.0,
                                                      6.0,
                                                      -7.0]))[0]
            df.loc[index_no_college, f"6138-{session}.{i}"] = 0.0
            # Find No answered
            # And Replace with 'NaN' value:
            index_no_answer = np.where(
                df.loc[:,f"6138-{session}.{i}"] == -3.0)[0]
            df.loc[index_no_answer, f"6138-{session}.{i}"] = np.NaN
        # Calculate Maximum Qualification
        max_qualification = \
            df.loc[:,
                   df.columns.str.startswith(f"6138-{session}.")].max(axis=1)
        # Add new column for qualification with Maximum value
        df.loc[:, "qualification"] = max_qualification

        return df

###############################################################################
############################## PREPROCESS FEATURES ############################
# Preprocess features or Handling Outliers
# more meaningful insights and patterns for machine learning models.
###############################################################################
    def preprocess_behaviours(
        self,
        df,
    ):
        """Preprocess Behavioural Phenotypes
      
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
        assert isinstance(session, int), "session must be a int!"
        # -----------------------------------------------------------
        # -------  Phynotypes without Data-Coding -------
        # The following behavioural don't have Data-Coding:
        # -------  Fluid intelligence task -------
        # '20016', Fluid intelligence score
        # -----------------------------------------------
        # -------  Reaction Time task -------
        # '20023', Mean time to correctly identify matches
        # -----------------------------------------------
        # -------  Pairs matching task -------
        #  A value of 0 indicates the participant made no mistakes. 
        # '399',  # Number of incorrect matches in round
        # -----------------------------------------------
        # -------  Trail making task -------
        # cognitive in clinic
        # Data-Coding: 1990
        #           0 --> Trail not completed
        # '6348',  # Duration to complete numeric path (trail #1) 
        # '6350',  # Duration to complete alphanumeric path (trail #2)
        # ------------------------------------
        # cognitive online --> No data-Coding
        # '20156',  # Duration to complete numeric path (trail #1)
        # '20157',  # Duration to complete alphanumeric path (trail #2)
        # ------------------------------------
        # I used '20156' and '20157' in place of '6348' and '6350'
        # Because '6348' and '6350' tasks taken only for MRI visits(instance 2&3)
        # And non-MRI healthy data don't contain this task.
        # '20156' and '20157' not contain 0 value.
        # -----------------------------------------------
        # ------- symbol digit matches -------
        # cognitive in clinic
        # Data-Coding: 6361
        #           0	Did not make any correct matches
        # '23323',  # Number of symbol digit matches attempted --> Data-Coding: 6361
        #           Number of symbol digit matches attempted before the participant timed-out.
        # '23324',  # Number of symbol digit matches made correctly --> No Data-Coding
        #           This is the number of symbols correctly matched to digits.
        # ------------------------------------
        # cognitive online --> No data-Coding
        # '20195',  # Number of symbol digit matches attempted
        # '20159',  # Number of symbol digit matches made correctly
        # ------------------------------------
        # I used '20195' and '20159' in place of '6348' and '6350'
        # Because '23323' and '23324' tasks taken only for MRI visits(instance 2&3)
        # And non-MRI healthy data don't contain this task.
        
        #######################################################
        # -------  Numeric memory task -------
        # '4282',  # Maximum digits remembered correctly
        # Data-Coding: 100696
        #           -1 --> Abandoned
        # ------------------------------------
        # Longest number correctly recalled during the numeric memory test. 
        # A value of -1 is recorded if the participant chose to abandon 
        # the test before completing the first round. So, Replace (-1) with NaN
        # ------------------------------------
        df.loc[:,f"4282-{session}.0"] = \
            df.loc[:,f"4282-{session}.0"].replace(-1, np.NaN)
        
        #######################################################
        # -------  Pairs matching task ---------
        # cognitive in clinic
        # '400',  # Time to complete round
        # Data-Coding: 402
        #           0 --> represents "Test not completed".
        # ------------------------------------
        # Defined-instances run from 0 to 3, 
        # Task ran 3 run on each instances:
        for i in range(1, 4):
            df.loc[:,f"400-{session}.{i}"] = \
                df.loc[:,f"400-{session}.{i}"].replace(0, np.NaN)
                
        #######################################################
        # ------- Prospective memory result -------
        # '20018',  # Prospective memory result
        # Data-Coding: 18
        #           0 --> Instruction not recalled, either skipped or incorrect
        #           1 --> Correct recall on first attempt
        #           2 --> Correct recall on second attempt
        # ------------------------------------
        # Replace value of 2 with 0 --> Based on Data-Coding 18 and BMC paper
        # Github link: https://github.com/Jiang-brain/
        #              Grip-strength-association/blob/
        #              main/association_grip_strength_behavior.m
        # Because:
        # this field condenses the results of the prospective memory test into 3 groups.
        # It does not distinguish between people who, at the first attempt, 
        # followed the on-screen instruction (thereby giving an incorrect result) and 
        # people who remembered to ignore the instruction 
        # but did not correctly recall what to do instead. 
        # ------------------------------------
        df.loc[:, f"20018-{session}.0"] = \
            df.loc[:, f"20018-{session}.0"].replace(2, 0)
        
        #######################################################
        # ------- Life satisfaction -------
        # Data-Coding: 100478
        #           1	Extremely happy
        #           2	Very happy
        #           3	Moderately happy
        #           4	Moderately unhappy
        #           5	Very unhappy
        #           6	Extremely unhappy
        #           -1	Do not know
        #           -3	Prefer not to answer
        # ------------------------------------
        # Data-Coding: 100479
        #           1	Extremely happy
        #           2	Very happy
        #           3	Moderately happy
        #           4	Moderately unhappy
        #           5	Very unhappy
        #           6	Extremely unhappy
        #           7	I am not employed
        #           -1	Do not know
        #           -3	Prefer not to answer
        # ------------------------------------
        # '4526',  # Happiness --> Data-Coding 100478
        # '4559',  # Family relationship satisfaction --> Data-Coding 100478
        # '4537',  # Work/job satisfaction --> Data-Coding 100479
        # '4548',  # Health satisfaction --> Data-Coding 100478
        # '4570',  # Friendships satisfaction --> Data-Coding 100478
        # '4581',  # Financial situation satisfaction --> Data-Coding 100478
        # ------------------------------------
        # Replace Happiness less than 0
        # with NaN based on Data-Coding 100478
        index = np.where(df.loc[:, f"4526-{session}.0"] < 0)[0]
        df.loc[index, f"4526-{session}.0"] = np.NaN
        # ------------------------------------
        # Replace Family relationship satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4559-{session}.0"] < 0)[0]
        df.loc[index, f"4559-{session}.0"] = np.NaN
        # ------------------------------------
        # https://github.com/Jiang-brain/Grip-strength-association/blob/3d3952ffb661e5e8a774b397f428a43dbe58f665/association_grip_strength_behavior.m#L75
        # Replace Work/job satisfaction less than 0 and more than 6
        # with NaN based on Data-Coding 100479
        index = np.where(
            (df.loc[:, f"4537-{session}.0"] < 0) | \
                (df.loc[:, f"4537-{session}.0"] > 6))[0]
        df.loc[index, f"4537-{session}.0"] = np.NaN
        # ------------------------------------
        # Replace Health satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4548-{session}.0"] < 0)[0]
        df.loc[index, f"4548-{session}.0"] = np.NaN
        # ------------------------------------
        # Replace Friendships satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4570-{session}.0"] < 0)[0]
        df.loc[index, f"4570-{session}.0"] = np.NaN
        # ------------------------------------
        # Replace Financial situation satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4581-{session}.0"] < 0)[0]
        df.loc[index, f"4581-{session}.0"] = np.NaN
        
        #######################################################
        # ------- Subjective well-being -------
        # Data-Coding: 537
        #           -818	Prefer not to answer
        #           -121	Do not know
        #           1	Extremely happy
        #           2	Very happy
        #           3	Moderately happy
        #           4	Moderately unhappy
        #           5	Very unhappy
        #           6	Extremely unhappy
        # ------------------------------------
        # Data-Coding: 538
        #           -818	Prefer not to answer
        #           -121	Do not know
        #           1	Not at all
        #           2	A little
        #           3	A moderate amount
        #           4	Very much
        #           5	An extreme amount
        # ------------------------------------
        # '20458',  # General happiness --> Data-Coding 537. 
        # '20459',  # General happiness with own health --> Data-Coding 537.
        # '20460',  # Belief that own life is meaningful --> Data-Coding 538
        # ------------------------------------
        # Replace General happiness less than 0
        # with NaN based on Data-Coding 537    
        index = np.where(df.loc[:, f"20458-{session}.0"] < 0)[0]
        df.loc[index, f"20458-{session}.0"] = np.NaN
        # ------------------------------------
        # Replace General happiness with own health less than 0
        # with NaN based on Data-Coding 537    
        index = np.where(df.loc[:, f"20459-{session}.0"] < 0)[0]
        df.loc[index, f"20459-{session}.0"] = np.NaN
        # ------------------------------------
        # Replace Belief that own life is meaningful less than 0
        # with NaN based on Data-Coding 538   
        index = np.where(df.loc[:, f"20460-{session}.0"] < 0)[0]
        df.loc[index, f"20460-{session}.0"] = np.NaN

        #######################################################
        
        return df

###############################################################################
############################## Remove NaN coulmns #############################
# Remove columns if their values are all NAN
###############################################################################
# Remove columns that all values are NaN
    def remove_nan_columns(
        self,
        df,
    ):
        """Remove columns with all NAN values
      
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
        assert isinstance(session, int), "session must be a int!"

        nan_cols = df.columns[df.isna().all()].tolist()
        df = df.drop(nan_cols, axis=1)
        
        return df
    
###############################################################################
def run_healthy_preprocessing(data, data_processor=None):
    if data_processor is None:
        # Create an instance of DataPreprocessor with session=0
        data_processor = DataPreprocessor(data,session=0)

    # Call all functions inside the class
    # DATA VALIDATION
    data = data_processor.validate_handgrips(data)
    # FEATURE ENGINEERING
    data = data_processor.calculate_qualification(data)
    data = data_processor.calculate_waist_to_hip_ratio(data)
    data = data_processor.calculate_neuroticism_score(data)
    data = data_processor.calculate_anxiety_score(data)
    data = data_processor.calculate_depression_score(data)
    data = data_processor.calculate_cidi_score(data)
    data = data_processor.preprocess_behaviours(data)
    data = data_processor.sum_handgrips(data)
    data = data_processor.calculate_left_hgs(data)
    data = data_processor.calculate_right_hgs(data)
    data = data_processor.sub_handgrips(data)
    data = data_processor.calculate_laterality_index(data)
    data = data_processor.remove_nan_columns(data)

    return data