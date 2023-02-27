#!/usr/bin/env Disorderspredwp3

"""
Add different columns based on corresponding Field-IDs, conditions to data.

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

# from ptpython.repl import embed


###############################################################################
class CalculatingNewColumns:
    def __init__(
        self,
        df: pd.DataFrame,
        session,
    ):
        """Calculate and add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df
        self.session = session

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
        df : df
            with extra column for:
            Waist to Hip Ratio
        """
        session = self.session

        assert isinstance(df, pd.df), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # Waist circumference field-ID: 48
        # Hip circumference field-ID: 49
        df[f"waist_to_hip_ratio-{session}.0"] = \
            (df[f"48-{session}.0"].astype(str).astype(float)).div(
                df[f"49-{session}.0"].astype(str).astype(float))

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47
        # sum of Handgrips (Left + Right)
        df[f"hgs(L+R)-{session}.0"] = \
            df[f"46-{session}.0"] + df[f"47-{session}.0"]

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47
        # Subtraction of Handgrips (Left - Right)
        df[f"hgs(L-R)-{session}.0"] = \
            df[f"46-{session}.0"] - df[f"47-{session}.0"]

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47

        df_sub = self.sub_handgrips(df)
        df_sum = self.sum_handgrips(df)

        # Laterality Index of Handgrip strength: (Left - Right)/(Left +right)
        df[f"hgs(LI)-{session}.0"] = \
            (df_sub[f"hgs(L-R)-{session}.0"] / df_sum[f"hgs(L+R)-{session}.0"])

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # hgs_left field-ID: 46
        # hgs_right field-ID: 47

        # ------- Handedness Field-ID: 1707
        # Data-Coding 100430

        # If handedness is equal to 1
        # Right hand is Dominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 1.0)[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[f"47-{session}.0"]
        # If handedness is equal to 2
        # Left hand is Dominant
        index = np.where(df.loc[:, f"1707-{session}.0"] == 2.0)[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[f"46-{session}.0"]
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer)
        # Dominant will be the Highest Handgrip score from both hands.
        index = np.where(
            (df.loc[:, f"1707-{session}.0"] == 3.0) |
            (df.loc[:, f"1707-{session}.0"] == -3.0))[0]
        df.loc[index, f"dominant_hgs-{session}.0"] = \
            df.loc[[f"46-{session}.0"], [f"47-{session}.0"]].max(axis=1)

        return df

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
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

        assert isinstance(df, pd.df), "df must be a dataframe!"
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
        session_string = f"-{session}.0"
        depression_fields = [s + session_string for s in depression_fields]

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
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

        session_string = f"-{session}.0"
        anxiety_fields = [s + session_string for s in anxiety_fields]

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
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
        session_string = f"-{session}.0"
        cidi_core_fields = [s + session_string for s in cidi_core_fields]
        cidi_noncore_fields = [s + session_string for s in cidi_noncore_fields]

        core_score = df.where(
            df[cidi_core_fields] >= 0.0).sum(axis=1, min_count=1)

        index = np.where(
            df.loc[:, f"20536-{session}.0"].isin([1.0, 2.0, 3.0]))[0]
        df.loc[index, f"20536-{session}.0"] = 1.0

        noncore_score = df.where(
            df[cidi_noncore_fields] >= 0.0).sum(axis=1, min_count=1)

        # Add 2 series with fill_value parameter for
        # any non NaN value be successful.
        cidi_score = core_score.add(noncore_score, fill_value=0)

        df.loc[:, "CIDI_score"] = cidi_score

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

        assert isinstance(df, pd.df), "df must be a dataframe!"
        assert isinstance(session, int), "session must be a int!"

        # ------- Qualification Field-ID: 6138
        # Data-Coding 100305
        max_qualification = \
            df.loc[:,
                   df.columns.str.startswith(f"6138-{session}.")].max(axis=1)
        df.loc[:, "qualification"] = max_qualification

        return df
