#!/usr/bin/env Disorderspredwp3

"""
Add different columns based on corresponding Field-IDs, conditions to data.

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

from ptpython.repl import embed


###############################################################################
class PreprocessData:
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
        
###############################################################################
    def validate_handgrips(
        self,
        df,
    ):
        session = self.session
        
        # If handedness is equal to 1
        # Right hand is Dominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 1.0)[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, f"47-{session}.0"]
        # If handedness is equal to 2
        # Left hand is Dominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, f"46-{session}.0"]
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer)
        # Dominant will be the Highest Handgrip score from both hands.
        index = np.where(
            (df.loc[:, f"1707-{session}.0"] == 3.0) |
            (df.loc[:, f"1707-{session}.0"] == -3.0)|
            (df.loc[:, f"1707-{session}.0"].isna()))[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, [f"46-{session}.0", f"47-{session}.0"]].max(axis=1)

        # If handedness is equal to 1
        # Left hand is nonDominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 1.0)[0]
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[:, f"46-{session}.0"]
        # If handedness is equal to 2
        # Right hand is nonDominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[:, f"47-{session}.0"]
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer)
        # NonDominant will be the Lowest Handgrip score from both hands.
        index = np.where(
            (df.loc[:, f"1707-{session}.0"] == 3.0) |
            (df.loc[:, f"1707-{session}.0"] == -3.0) |
            (df.loc[:, f"1707-{session}.0"].isna()))[0]
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[:, [f"46-{session}.0", f"47-{session}.0"]].min(axis=1)

        df = df[df.loc[:, f"dominant_hgs-{session}.0"] >=4].reset_index()

        return df
    
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
            with extra column for:
            Waist to Hip Ratio
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # Waist circumference field-ID: 48
        # Hip circumference field-ID: 49
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
            with calculating extra column for:
            (HGS Left + HGS Right)
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47
        # sum of Handgrips (Left + Right)
        df.loc[:, f"hgs(L+R)-{session}.0"] = \
            df.loc[:, f"46-{session}.0"] + df.loc[:, f"47-{session}.0"]

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
            with calculating extra column for:
            (HGS Left - HGS Right)
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

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
    def dominant_handgrip(
        self,
        df,
    ):
        """Extract dominant handgrip
        and add "dominant_hgs" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for:
            Dominant hand Handgrip strength
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47

        # ------- Handedness Field-ID: 1707
        # Data-Coding 100430

        # If handedness is equal to 1
        # Right hand is Dominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 1.0)[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, f"47-{session}.0"]
        # If handedness is equal to 2
        # Left hand is Dominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[:, f"46-{session}.0"]
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer)
        # Dominant will be the Highest Handgrip score from both hands.
        index = np.where(
            (df.loc[:, f"1707-{session}.0"] == 3.0) |
            (df.loc[:, f"1707-{session}.0"] == -3.0))[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df[[f"46-{session}.0", f"47-{session}.0"]].max(axis=1)

###############################################################################
    def nondominant_handgrip(
        self,
        df,
    ):
        """Extract Non-dominant handgrip
        and add "nondominant_hgs" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for:
            NonDominant hand Handgrip strength
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47

        # ------- Handedness Field-ID: 1707
        # Data-Coding 100430

        # If handedness is equal to 1
        # Left hand is nonDominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 1.0)[0]
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[f"46-{session}.0"]
        # If handedness is equal to 2
        # Right hand is nonDominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[f"47-{session}.0"]
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer)
        # NonDominant will be the Lowest Handgrip score from both hands.
        index = np.where(
            (df.loc[:, f"1707-{session}.0"] == 3.0) |
            (df.loc[:, f"1707-{session}.0"] == -3.0))[0]
        df.loc[index, f"nondominant_hgs-{session}.0"] = \
            df.loc[[f"46-{session}.0"],
                   [f"47-{session}.0"]].min(axis=1)

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
            with extra column for:
            neuroticism score
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        neuroticism_fields = [
            # Mood swings
            # Data-Coding: 100349
            '1920',
            # Miserableness
            # Data-Coding: 100349
            '1930',
            # Irritability
            # Data-Coding: 100349
            '1940',
            # Sensitivity / hurt feelings
            # Data-Coding: 100349
            '1950',
            # Fed-up feelings
            # Data-Coding: 100349
            '1960',
            # Nervous feelings
            # Data-Coding: 100349
            '1970',
            # Worrier / anxious feelings
            # Data-Coding: 100349
            '1980',
            # Tense / 'highly strung'
            # Data-Coding: 100349
            '1990',
            # Worry too long after embarrassment
            # Data-Coding: 100349
            '2000',
            # Suffer from 'nerves'
            # Data-Coding: 100349
            '2010',
            # Loneliness, isolation
            # Data-Coding: 100349
            '2020',
            # Guilty feelings
            # Data-Coding: 100349
            '2030',
        ]

        neuroticism_fields = [s + f"-{session}.0" for s in neuroticism_fields]
        
        df.loc[:, "neuroticism_score"] = \
            df.where(df.loc[:, neuroticism_fields] >= 0.0).sum(axis=1, min_count=9)

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
            with extra column for:
            depression score
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        depression_fields = [
            # Recent feelings of inadequacy
            # Data-Coding 504
            '20507',
            # trouble concentrating on things
            # Data-Coding 504
            '20508',
            # Recent feelings of depression
            # Data-Coding 504
            '20510',
            # Recent poor appetite or overeating
            # Data-Coding 504
            '20511',
            # Recent thoughts of suicide or self-harm
            # Data-Coding 504
            '20513',
            # Recent lack of interest or pleasure in doing things
            # Data-Coding 504
            '20514',
            # Trouble falling or staying asleep, or sleeping too much
            # Data-Coding 504
            '20517',
            # Recent changes in speed/amount of moving or speaking
            # Data-Coding 504
            '20518',
            # Recent feelings of tiredness or low energy
            # Data-Coding 504
            '20519',
        ]

        depression_fields = [s + f"-{session}.0" for s in depression_fields]

        df.loc[:, "depression_score"] = df.where(
            df[depression_fields] > 0.0).sum(axis=1, min_count=1)

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
            with extra column for:
            anxiety score
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        anxiety_fields = [
            # Recent easy annoyance or irritability
            # Data-Coding 504
            '20505',
            # Recent feelings or nervousness or anxiety
            # Data-Coding 504
            '20506',
            # Recent inability to stop or control worrying
            # Data-Coding 504
            '20509',
            # Recent feelings of foreboding
            # Data-Coding 504
            '20512',
            # Recent trouble relaxing
            # Data-Coding 504
            '20515',
            # Recent restlessness -> Data-Coding 504
            '20516',
            # Recent worrying too much about different things
            # Data-Coding 504
            '20520',
        ]

        anxiety_fields = [s + f"-{session}.0" for s in anxiety_fields]

        df.loc[:, "anxiety_score"] = \
            df.where(df[anxiety_fields] > 0.0).sum(axis=1, min_count=1)

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
            with extra column for:
            CIDI score
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # ------- CIDI core symptyms of depression
        cidi_core_fields = [
            # Ever had prolonged loss of interest in normal activities
            # Data-Coding 503
            '20441',
            # Ever had prolonged feelings of sadness or depression
            # Data-Coding 503
            '20446',
        ]
        # ------- CIDI non-core symptyms of depression
        cidi_noncore_fields = [
            # Feelings of tiredness during worst
            # episode of depression
            # Data-Coding 502
            '20449',
            # Feelings of worthlessness during worst
            # period of depression
            # Data-Coding 502
            '20450',
            # Thoughts of death during worst depression
            #  Data-Coding 502
            '20437',
            # Difficulty concentrating during worst depression
            # Data-Coding 502
            '20435',
            # Did your sleep change?
            # Data-Coding 502
            '20532',
            # Weight change during worst episode of depression
            # Data-Coding 507
            '20536',
        ]

        cidi_core_fields = [s + f"-{session}.0" for s in cidi_core_fields]
        cidi_noncore_fields = [s +
                               f"-{session}.0" for s in cidi_noncore_fields]
        # print("============================ Done! ============================")
        # embed(globals(), locals())
        index = np.where(
            df.loc[:, f"20536-{session}.0"].isin([1.0, 2.0, 3.0]))[0]
        df.loc[index, f"20536-{session}.0"] = 1.0
        # df.where is replacing all negative values with NaN
        core_score = df.where(
            df[cidi_core_fields] >= 0.0).sum(axis=1, min_count=1)

        core_score = core_score.replace(2.0, 1.0)

        # df.where is replacing all negative values with NaN
        noncore_score = df.where(
            df[cidi_noncore_fields] >= 0.0).sum(axis=1, min_count=4)
        # -----> change min_count=4

        # Add 2 series with fill_value parameter for
        # any non NaN value be successful.
        cidi_score = core_score.add(noncore_score, fill_value=0)

        # if cidi_score is nan or core_score is nan then cidi_score =np.NaN
        index = np.where(cidi_score.isna() | core_score.isna())[0]
        cidi_score.loc[index] = np.NaN
        
        # Yes so they asked these questions only when one of the core was answered yes
        # So for those subjects that answered both core questions as no (i.e. 0) we use cidi score 0.
        # Please make a note of this in the code

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
            with extra column for:
            maximum qualification
        """
        session = self.session

        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"
        # -------  Qualification -------
        # '6138',	  # Qualifications
        # Data-coding: 100305
        for i in range(6):
            index_college = np.where(
                df.loc[:,f"6138-{session}.{i}"] == 1.0)[0]
            df.loc[index_college, f"6138-{session}.{i}"] = 1.0
            
            index_no_college = np.where(
                df.loc[:,f"6138-{session}.{i}"].isin([2.0,
                                                      3.0,
                                                      4.0,
                                                      5.0,
                                                      6.0,
                                                      -7.0]))[0]
            df.loc[index_no_college, f"6138-{session}.{i}"] = 0.0
            
            index_no_answer = np.where(
                df.loc[:,f"6138-{session}.{i}"] == -3.0)[0]
            df.loc[index_no_answer, f"6138-{session}.{i}"] = np.NaN
        
        max_qualification = \
            df.loc[:,
                   df.columns.str.startswith(f"6138-{session}.")].max(axis=1)
        df.loc[:, "qualification"] = max_qualification

        return df

###############################################################################
    def preprocess_behaviours(
        self,
        df,
    ):
        session = self.session

        # -------  Numeric memory task -------
        # '4282',  # Maximum digits remembered correctly --> Data-Coding: 100696
        # Replace (-1) with NaN
        # A value of -1 is recorded if the participant chose
        # to abandon the test before completing the first round. 
        df.loc[:,f"4282-{session}.0"] = df.loc[:,f"4282-{session}.0"].replace(-1, np.NaN)
        
        #######################################################
        # -------  Trail making task -------
        # cognitive in clinic -> Data-Coding 1990
        # '6348',  # Duration to complete numeric path (trail #1) 
        # '6350',  # Duration to complete alphanumeric path (trail #2)
        # cognitive online
        # '20156',  # Duration to complete numeric path (trail #1)
        # '20157',  # Duration to complete alphanumeric path (trail #2)
        # 0	--> Trail not completed
        df.loc[:,f"20156-{session}.0"] = df.loc[:,f"20156-{session}.0"].replace(0, np.NaN)
        df.loc[:,f"20157-{session}.0"] = df.loc[:,f"20157-{session}.0"].replace(0, np.NaN)
        
        #######################################################
        # ------- Prospective memory result -------
        # '20018',  # Prospective memory result --> Data-Coding 18
        # Replace value of 2 with 0
        # Based on Data-Coding 18 and BMC paper
        df.loc[:, f"20018-{session}.0"] = df.loc[:, f"20018-{session}.0"].replace(2, 0)
        
        #######################################################
        # ------- Life satisfaction -------
        # '4526',  # Happiness --> Data-Coding 100478.
        # '4559',  # Family relationship satisfaction --> Data-Coding 100478.
        # '4537',  # Work/job satisfaction --> Data-Coding 100479.
        # '4548',  # Health satisfaction --> Data-Coding 100478.
        # '4570',  # Friendships satisfaction --> Data-Coding 100478.
        # '4581',  # Financial situation satisfaction --> Data-Coding 100478.
        
        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 100478
        index = np.where(df.loc[:, f"4526-{session}.0"] < 0)[0]
        df.loc[index, f"4526-{session}.0"] = np.NaN
        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4559-{session}.0"] < 0)[0]
        df.loc[index, f"4559-{session}.0"] = np.NaN
        
        # Replace job satisfaction less than 0 and more than 6
        # with NaN based on Data-Coding 100479
        index = np.where(
            (df.loc[:, f"4537-{session}.0"] < 0) | \
                (df.loc[:, f"4537-{session}.0"] > 6))[0]
        df.loc[index, f"4537-{session}.0"] = np.NaN
        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4548-{session}.0"] < 0)[0]
        df.loc[index, f"4548-{session}.0"] = np.NaN
        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4570-{session}.0"] < 0)[0]
        df.loc[index, f"4570-{session}.0"] = np.NaN
        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 100478    
        index = np.where(df.loc[:, f"4581-{session}.0"] < 0)[0]
        df.loc[index, f"4581-{session}.0"] = np.NaN
        
        #######################################################
        # ------- Subjective well-being -------
        # '20458',  # General happiness --> Data-Coding 537. 
        # '20459',  # General happiness with own health --> Data-Coding 537.
        # '20460',  # Belief that own life is meaningful --> Data-Coding 538
        
        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 537    
        index = np.where(df.loc[:, f"20458-{session}.0"] < 0)[0]
        df.loc[index, f"20458-{session}.0"] = np.NaN

        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 537    
        index = np.where(df.loc[:, f"20459-{session}.0"] < 0)[0]
        df.loc[index, f"20459-{session}.0"] = np.NaN
        
        # Replace job satisfaction less than 0
        # with NaN based on Data-Coding 538   
        index = np.where(df.loc[:, f"20460-{session}.0"] < 0)[0]
        df.loc[index, f"20460-{session}.0"] = np.NaN

        #######################################################
        # -------  Pairs matching task ---------
                # cognitive in clinic
        # '399',  # Number of incorrect matches in round
        #  A value of 0 indicates the participant made no mistakes. 
        for i in range(1, 4):
            df.loc[:,f"399-{session}.{i}"] = \
                df.loc[:,f"399-{session}.{i}"].replace(0, np.NaN)
        
        # '400',  # Time to complete round --> Data-Coding: 402
        # A value of 0 represents "Test not completed".
        for i in range(1, 4):
            df.loc[:,f"400-{session}.{i}"] = \
                df.loc[:,f"400-{session}.{i}"].replace(0, np.NaN)
        
        # ------- symbol digit matches -------
        # cognitive in clinic
        # '23323',  # Number of symbol digit matches attempted --> Data-Coding: 6361
        # '23324',  # Number of symbol digit matches made correctly
        # cognitive online
        # '20195',  # Number of symbol digit matches attempted
        # '20159',  # Number of symbol digit matches made correctly
        # A value of 0 represents "Did not make any correct matches".
        df.loc[:,f"20195-{session}.0"] = df.loc[:,f"20195-{session}.0"].replace(0, np.NaN)
        
        return df
