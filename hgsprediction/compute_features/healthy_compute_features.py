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
# This class extract all required features from data:
def compute_features(df, mri_status, feature_type, session):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(mri_status, str ), "mri_status must be a string!"
    assert isinstance(feature_type, str ), "feature_type must be a string!"

    # if mri_status == "nonmri":
    #     session = "0"
    # elif mri_status == "mri":
    #     session = "2"
    # -----------------------------------------------------------
    df = calculate_age(df, session)
    df = calculate_gender(df, session)
    df = calculate_handedness(df, session)

    if feature_type == "anthropometrics":
        df = calculate_anthropometrics(df, session)
                    
    elif feature_type == "behavioral":
        df = calculate_behavioral(df, session)

    elif feature_type == "qualification":
        df = calculate_qualification(df, session)
        
    elif feature_type == "socioeconomic_status":
        df = calculate_socioeconomic_status(df, session)

    return df

###############################################################################
############################## FFEATURE ENGINEERING ###########################
# Creating new columns/features/targets from existing data
# Preprocess features or Handling Outliers
# more meaningful insights and patterns for machine learning models.
###############################################################################
def calculate_anthropometrics(df, session):
    df = calculate_bmi(df, session)
    df = calculate_height(df, session)
    df = calculate_WHR(df, session)
    
    return df
###############################################################################
def calculate_behavioral(df, session):
    # Totally 25 fields:
    # (N=12)
    df = calculate_cognitive_functioning(df, session)
    # (N=4)
    df = calculate_depression_score(df, session)
    df = calculate_anxiety_score(df, session)
    df = calculate_cidi_score(df, session)
    df = calculate_neuroticism_score(df, session)
    # (N=6)
    df = calculate_life_satisfaction(df, session)
    # (N=3)
    df = calculate_well_being(df, session)
    
    return df
###############################################################################
def calculate_bmi(df, session):
    """Calculate coressponding BMI
    and add "BMI" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: BMI
    """    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    # Add a new column 'new_column'
    bmi = f"bmi-{session}.0" 
    df.loc[:, bmi] = df.loc[:, f"21001-{session}.0"]

    return df

###############################################################################
def calculate_height(df, session):
    """Calculate coressponding Height
    and add "Height" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: Height
    """
    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    height = f"height-{session}.0"
    df.loc[:, height] = df.loc[:, f"50-{session}.0"]

    return df

###############################################################################
def calculate_WHR(df, session):
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

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"
    # ------------------------------------
    # Add new column "WHR" by the following process: 
    # Waist circumference field-ID: 48
    # Hip circumference field-ID: 49
    # Calculating Waist/Hip
    whr = f"waist_to_hip_ratio-{session}.0" 
    df.loc[:, whr] = df.loc[:, f"48-{session}.0"]/df.loc[:, f"49-{session}.0"]

    return df

###############################################################################
def calculate_age(df, session):
    """Calculate coressponding Age
    and add "Age" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: Age
    """
    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    if session == "0":
        df.loc[:, "age"] = df.loc[:, "Age1stVisit"]
    elif session == "2":
        df.loc[:, "age"] = df.loc[:, "AgeAtScan"]
    elif session == "3":
        df.loc[:, "age"] = df.loc[:, "AgeAt2ndScan"]
    return df    

###############################################################################
def calculate_gender(df, session):
        """Calculate coressponding gender
        and add "gender" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: WHR
        """
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session, str), "session must be a string!"
            # -----------------------------------------------------------
        df.loc[:, "gender"] = df.loc[:, "31-0.0"]
        
        return df

###############################################################################
def calculate_handedness(df, session):
    """Calculate dominant handgrip
    and add "handedness" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: Dominant hand Handgrip strength
    """
    # session = self.session
    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    
    # Add a new column 'new_column'
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
    if session in ["0", "3"]:
        # Add and new column "handedness"
        # And assign Right hand HGS value
        df.loc[df["1707-0.0"] == 1.0, handedness] = 1.0
        # If handedness is equal to 2
        # Right hand is Non-Dominant
        # Find handedness equal to 2:
        # Add and new column "handedness"
        # And assign Left hand HGS value:  
        df.loc[df["1707-0.0"] == 2.0, handedness] = 2.0
        # ------------------------------------
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer) OR
        # NaN value
        # Dominant will be the Highest Handgrip score from both hands.
        # Find handedness equal to 3, -3 or NaN:
        # Add and new column "handedness"
        # And assign Highest HGS value among Right and Left HGS:
        # Add and new column "handedness"
        # And assign lowest HGS value among Right and Left HGS:
        df.loc[df["1707-0.0"].isin([3.0, -3.0, np.nan]), handedness] = 3.0
        
    elif session == "2":
        index = df[df.loc[:, "1707-2.0"] == 1.0].index
        df.loc[index, handedness] = 1.0
        index = df[df.loc[:, "1707-2.0"] == 2.0].index
        df.loc[index, handedness] = 2.0
            
        index = df[df.loc[:, "1707-2.0"].isin([3.0, -3.0, np.NaN])].index
        filtered_df = df.loc[index, :]
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 1.0].index
        df.loc[inx, handedness] = 1.0
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 2.0].index
        df.loc[inx, handedness] = 2.0
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"].isin([3.0, -3.0, np.NaN])].index
        df.loc[inx, handedness] = 3.0

    return df

###############################################################################
def calculate_neuroticism_score(df, session):
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
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"
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
def calculate_depression_score(df, session):
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
    # The only aailable session for Depression is 0:
    session = "0"

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"
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
def calculate_anxiety_score(df, session):
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
    # The only aailable session for Anxiety is 0:
    session = "0"

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"
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
def calculate_cidi_score(df, session):
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
    # The only aailable session for CIDI is 0:
    session = "0"

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"

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
############################## PREPROCESS FEATURES ############################
# Preprocess features or Handling Outliers
# more meaningful insights and patterns for machine learning models.
###############################################################################
def calculate_life_satisfaction(df, session):

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
    # '4548',  # Health satisfaction --> Data-Coding 100478
    # '4559',  # Family relationship satisfaction --> Data-Coding 100478
    # '4570',  # Friendships satisfaction --> Data-Coding 100478
    # '4581',  # Financial situation satisfaction --> Data-Coding 100478
    # '4537',  # Work/job satisfaction --> Data-Coding 100479
    # '4526',  # Happiness --> Data-Coding 100478
    # ------------------------------------
    # Replace Health satisfaction less than 0
    # with NaN based on Data-Coding 100478
    # feild (1)
    health_satisfaction = f"health_satisfaction-{session}.0"
    df.loc[:, health_satisfaction] =  df.loc[:, f"4548-{session}.0"]      
    # index = df[df.loc[:, f"4548-{session}.0"] < 0].index
    index = df[df.loc[:, health_satisfaction] < 0].index
    df.loc[index, health_satisfaction] = np.NaN
    # ------------------------------------
    # Replace Family relationship satisfaction less than 0
    # with NaN based on Data-Coding 100478   
    # feild (2)    
    family_satisfaction = f"family_satisfaction-{session}.0" 
    df.loc[:, family_satisfaction] =  df.loc[:, f"4559-{session}.0"]    
    index = df[df.loc[:, family_satisfaction] < 0].index
    df.loc[index, family_satisfaction] = np.NaN 
    # ------------------------------------
    # Replace Friendships satisfaction less than 0
    # with NaN based on Data-Coding 100478    
    # feild (3)
    friendship_satisfaction = f"friendship_satisfaction-{session}.0"
    df.loc[:, friendship_satisfaction] =  df.loc[:, f"4570-{session}.0"]    
    index = df[df.loc[:, friendship_satisfaction] < 0].index
    df.loc[index, friendship_satisfaction] = np.NaN
    # ------------------------------------
    # Replace Financial situation satisfaction less than 0
    # with NaN based on Data-Coding 100478    
    # feild (4)
    financial_satisfaction = f"financial_satisfaction-{session}.0"
    df.loc[:, financial_satisfaction] =  df.loc[:, f"4581-{session}.0"]    
    index = df[df.loc[:, financial_satisfaction] < 0].index        
    df.loc[index, financial_satisfaction] = np.NaN
    # ------------------------------------
    # https://github.com/Jiang-brain/Grip-strength-association/blob/3d3952ffb661e5e8a774b397f428a43dbe58f665/association_grip_strength_behavior.m#L75
    # Replace Work/job satisfaction less than 0 and more than 6
    # with NaN based on Data-Coding 100479
    # feild (5)
    job_satisfaction = f"job_satisfaction-{session}.0"
    df.loc[:, job_satisfaction] =  df.loc[:, f"4537-{session}.0"]    
    index = df[(df.loc[:, job_satisfaction] < 0) | (df.loc[:, job_satisfaction] > 6)].index        
    df.loc[index, job_satisfaction] = np.NaN
    # ------------------------------------
    # Replace Happiness less than 0
    # with NaN based on Data-Coding 100478
    # feild (6)
    happiness = f"happiness-{session}.0"
    df.loc[:, happiness] =  df.loc[:, f"4526-{session}.0"]    
    index = df[df.loc[:, happiness] < 0].index
    df.loc[index, happiness] = np.NaN
    
    return df
###############################################################################
def calculate_well_being(df, session):
    ############ Subjective well-being ###############
    # Available only in session 0
    session = "0"
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
    # feild (1)
    general_happiness = f"general_happiness-{session}.0"
    df.loc[:, general_happiness] =  df.loc[:, f"20458-{session}.0"] 
    index = df[df.loc[:, general_happiness] < 0].index  
    # index = df[df.loc[:, f"20458-{session}.0"] < 0].index
    df.loc[index, general_happiness] = np.NaN
    # ------------------------------------
    # Replace happiness with own health less than 0
    # with NaN based on Data-Coding 537
    # feild (2)
    health_happiness = f"health_happiness-{session}.0"
    df.loc[:, health_happiness] =  df.loc[:, f"20459-{session}.0"]
    # index = df[df.loc[:, f"20459-{session}.0"] < 0].index
    index = df[df.loc[:, health_happiness] < 0].index
    df.loc[index, health_happiness] = np.NaN
    # ------------------------------------
    # Replace Belief that own life is meaningful less than 0
    # with NaN based on Data-Coding 538
    # feild (3) 
    belief_life_meaningful = f"belief_life_meaningful-{session}.0"
    df.loc[:, belief_life_meaningful] =  df.loc[:, f"20460-{session}.0"]
    # index = df[df.loc[:, f"20460-{session}.0"] < 0].index
    index = df[df.loc[:, belief_life_meaningful] < 0].index
    df.loc[index, belief_life_meaningful] = np.NaN
    
    return df

###############################################################################
def calculate_cognitive_functioning(df, session):
    """Preprocess Behavioural Phenotypes
    
    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
    """  
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"
    #######################################################
    ################## COGNITIVE FUNCTIONING ##############
    #######################################################
    # -------  Phynotypes without Data-Coding -------
    # The following behavioural don't have Data-Coding:
    # cognitive (1)
    # -------  Fluid intelligence task -------
    # '20016', Fluid intelligence score
    fluid_intelligence = f"fluid_intelligence-{session}.0"
    df.loc[:, fluid_intelligence] =  df.loc[:, f"20016-{session}.0"]
    # -----------------------------------------------
    # cognitive (2)
    # -------  Reaction Time task -------
    # '20023', Mean time to correctly identify matches
    reaction_time = f"reaction_time-{session}.0"
    df.loc[:, reaction_time] =  df.loc[:, f"20023-{session}.0"]
    #######################################################
    # cognitive (3)
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
    prospective_memory = f"prospective_memory-{session}.0"
    df.loc[:, prospective_memory] = df.loc[:, f"20018-{session}.0"].replace(2, 0)
    #######################################################
    # cognitive (4)
    # -------  Numeric memory task -------
    # '4282',  # Maximum digits remembered correctly
    # Data-Coding: 100696
    #           -1 --> Abandoned
    # ------------------------------------
    # Longest number correctly recalled during the numeric memory test. 
    # A value of -1 is recorded if the participant chose to abandon 
    # the test before completing the first round. So, Replace (-1) with NaN
    # ------------------------------------
    numeric_memory_Max_digits =  f"numeric_memory_Max_digits-{session}.0"
    df.loc[:, numeric_memory_Max_digits] =  df.loc[:,f"4282-{session}.0"].replace(-1, np.NaN)
    #######################################################
    # cognitive (5,6)
    # -------  Trail making task -------
    # MRI Sessions
    # cognitive in clinic
    # Data-Coding: 1990
    #           0 --> Trail not completed
    # '6348',  # Duration to complete numeric path (trail #1) 
    # '6350',  # Duration to complete alphanumeric path (trail #2)
    # ------------------------------------
    # non-MRI Sessions
    # cognitive online --> No data-Coding
    # '20156',  # Duration to complete numeric path (trail #1)
    # '20157',  # Duration to complete alphanumeric path (trail #2)
    # ------------------------------------
    # I used '20156' and '20157' in place of '6348' and '6350'
    # Because '6348' and '6350' tasks taken only for MRI visits(instance 2&3)
    # And non-MRI healthy data don't contain this task.
    # '20156' and '20157' not contain 0 value.
    # ------------------------------------
    # cognitive (5)
    trail_making_duration_numeric = f"trail_making_duration_numeric-{session}.0"
    # cognitive (6)
    trail_making_duration_alphanumeric = f"trail_making_duration_alphanumeric-{session}.0"
    if session == "0":
        df.loc[:,trail_making_duration_numeric] = df.loc[:,f"20156-{session}.0"].replace(0, np.NaN)
        df.loc[:,trail_making_duration_alphanumeric] = df.loc[:,f"20157-{session}.0"].replace(0, np.NaN)
    elif session == "2":
        df.loc[:, trail_making_duration_numeric] =  df.loc[:,f"6348-{session}.0"].replace(0, np.NaN)
        df.loc[:, trail_making_duration_alphanumeric] =  df.loc[:,f"6350-{session}.0"].replace(0, np.NaN)
    #######################################################
    # cognitive(7,8)
    # ------- symbol digit matches -------
    # MRI sessions
    # cognitive in clinic
    # Data-Coding: 6361
    #           0	Did not make any correct matches
    # '23323',  # Number of symbol digit matches attempted --> Data-Coding: 6361
    #           Number of symbol digit matches attempted before the participant timed-out.
    # '23324',  # Number of symbol digit matches made correctly --> No Data-Coding
    #           This is the number of symbols correctly matched to digits.
    # ------------------------------------
    # non-MRI sessions
    # cognitive online --> No data-Coding
    # '20195',  # Number of symbol digit matches attempted
    # '20159',  # Number of symbol digit matches made correctly
    # ------------------------------------
    # I used '20195' and '20159' in place of '6348' and '6350'
    # Because '23323' and '23324' tasks taken only for MRI visits(instance 2&3)
    # And non-MRI healthy data don't contain this task.
    # ------------------------------------
    # cognitive (7)
    symbol_digit_matches_attempted = f"symbol_digit_matches_attempted-{session}.0"
    # cognitive (8)
    symbol_digit_matches_corrected = f"symbol_digit_matches_corrected-{session}.0"
    if session == "0":
        df.loc[:, symbol_digit_matches_attempted] =  df.loc[:, f"20195-{session}.0"]
        df.loc[:, symbol_digit_matches_corrected] =  df.loc[:, f"20159-{session}.0"]
    elif session == "2":
        df.loc[:, symbol_digit_matches_attempted] =  df.loc[:, f"23323-{session}.0"]
        df.loc[:, symbol_digit_matches_corrected] =  df.loc[:, f"23324-{session}.0"]
    #######################################################
    # cognitive(9,10,11,12)
    # -------  Pairs matching task -------
    # Defined-instances run from 0 to 3, 
    # Task has 3 matches on each instances:
    # match 1 --> 3 pairs of symbol cards
    # match 2 --> 6 pairs of symbol cards 
    # match 3 --> 8 pairs of symbol cards
    # -----------------------------------------------
    #  A value of 0 indicates the participant made no mistakes.
    # Number of incorrect matches in round
    # This feild doen't have Data-coding
    # '399',  # Number of incorrect matches in round
    # -----------------------------------------------
    # cognitive in clinic
    # '400',  # Time to complete round
    # Data-Coding: 402
    #           0 --> represents "Test not completed".
    # ------------------------------------  
    # cognitive (9)
    pairs_matching_incorrected_number_3pairs = f"pairs_matching_incorrected_number_3pairs-{session}.0"
    # cognitive (10)
    pairs_matching_incorrected_number_6pairs = f"pairs_matching_incorrected_number_6pairs-{session}.0"
    # cognitive (11)
    pairs_matching_completed_time_3pairs = f"pairs_matching_completed_time_3pairs-{session}.0"
    # cognitive (12)
    pairs_matching_completed_time_6pairs = f"pairs_matching_completed_time_6pairs-{session}.0"
    
    # '399',  # Number of incorrect matches in round
    # *** (399-3) Not available for non-MRI and MRI.
    df.loc[:, pairs_matching_incorrected_number_3pairs] =  df.loc[:, f"399-{session}.1"]
    df.loc[:, pairs_matching_incorrected_number_6pairs] =  df.loc[:, f"399-{session}.2"]
    # '400',  # Time to complete round
    # *** (400-3) Not available for non-MRI and MRI.
    df.loc[:, pairs_matching_completed_time_3pairs] =  df.loc[:, f"400-{session}.1"].replace(0, np.NaN) 
    df.loc[:, pairs_matching_completed_time_6pairs] =  df.loc[:, f"400-{session}.2"].replace(0, np.NaN)
    #########################################################
    ############## Missing Feilds for non-MRI ##############
    ############# out of 30 behavioral features #############
    # (13,14) -------  Pairs matching task ---------
    # *** (399-3, 400-3) Not available for non-MRI and MRI.
    # *** Because, these tasks aren't available for non-MRI. 
    # *** So, ignored these tasks on MRI data analysis too.
    # -------------------------------------------------------          
    # (15) -------  Matrix pattern completion task -------
    # *** Not available for our non-MRI data
    # *** So, didn't use for MRI data too
    # '6373',  # Number of puzzles correctly solved
    # -------------------------------------------------------              
    # (16) -------  Tower rearranging task -------
    # *** Not available for our non-MRI data
    # *** So, didn't use for MRI data too                
    # '21004',  # Number of puzzles correct
    # -------------------------------------------------------          
    # (17) -------  Paired associate learning task -------
    # *** Not available for our non-MRI data
    # *** So, didn't use for MRI data too                
    # '20197',  # Number of word pairs correctly associated
    
    return df
    
###############################################################################
def calculate_qualification(df, session):
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
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"
    print("===== Done! =====")
    embed(globals(), locals())            
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
        # index_college = np.where(
        #     df.loc[:,f"6138-{session}.{i}"] == 1.0)[0]
        index_college = df[df.loc[:, f"6138-{session}.{i}"] == 1.0].index
        df.loc[index_college, f"6138-{session}.{i}"] = 1.0
        # Find No College or University degree
        # And Replace with '0.0' value:
        # index_no_college = np.where(
        #     df.loc[:,f"6138-{session}.{i}"].isin([2.0,
        #                                           3.0,
        #                                           4.0,
        #                                           5.0,
        #                                           6.0,
        #                                           -7.0]))[0]
        index_no_college = df[df.loc[:,f"6138-{session}.{i}"].isin([2.0,
                                                                    3.0,
                                                                    4.0,
                                                                    5.0,
                                                                    6.0,
                                                                    -7.0])].index         
        df.loc[index_no_college, f"6138-{session}.{i}"] = 0.0
        # Find No answered
        # And Replace with 'NaN' value:
        # index_no_answer = np.where(
        #     df.loc[:,f"6138-{session}.{i}"] == -3.0)[0]
        index_no_answer = df[df.loc[:,f"6138-{session}.{i}"] == -3.0].index
        df.loc[index_no_answer, f"6138-{session}.{i}"] = np.NaN
    # Calculate Maximum Qualification
    max_qualification = df.loc[:,df.columns.str.startswith(f"6138-{session}.")].max(axis=1)
    # Add new column for qualification with Maximum value
    df.loc[:, "qualification"] = max_qualification

    return df

###############################################################################
def calculate_socioeconomic_status(df, session):
    """Calculate Townsend deprivation index at recruitment score
    and add "Townsend deprivation index at recruitment score (TDI_score)" column
    to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: TDI_score
    """
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a str!"
    
    session = "0"
    # ------------------------------------
    # Townsend deprivation index at recruitment
    tdi_score = "TDI_score"
    df.loc[:, tdi_score] = df.loc[:, "22189-0.0"]
    
    return df
############################## Remove NaN coulmns #############################
# Remove columns if their values are all NAN
###############################################################################
# Remove columns that all values are NaN
def remove_nan_columns(df, session):
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
    assert isinstance(session, str), "session must be a str!"

    nan_cols = df.columns[df.isna().all()].tolist()
    df = df.drop(nan_cols, axis=1)
    
    return df

###############################################################################