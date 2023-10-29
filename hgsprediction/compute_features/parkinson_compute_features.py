
import pandas as pd
import numpy as np
from ptpython.repl import embed

###############################################################################
# This class extract all required features from data:
def compute_features(df, session_column, feature_type, mri_status):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str ), "session_column must be a string!"
    assert isinstance(feature_type, str ), "feature_type must be a string!"
    assert isinstance(mri_status, str ), "mri_status must be a string!"
    # -----------------------------------------------------------
    df = calculate_gender(df, session_column)
    df = calculate_age(df, session_column)
    df = calculate_handedness(df, session_column, mri_status)    
    df = calculate_days(df, session_column)
    df = calculate_years(df, session_column)

    if feature_type == "anthropometrics":
        df = calculate_anthropometrics(df, session_column)
                    
    # elif feature_type == "behavioral":
    #     df = calculate_behavioral(df, session_column)

    # elif feature_type == "qualification":
    #     df = calculate_qualification(df, session_column)
        
    # elif feature_type == "socioeconomic_status":
    #     df = calculate_socioeconomic_status(df, session_column)

    # elif feature_type == "behavioral":
    #     features = ca()   
                    
    # elif feature_type == "anthropometrics_behavioral_gender":
    #     features = extract_anthropometric_features() + extract_behavioral_features() + extract_gender_features()
            
    return df
###############################################################################
###############################################################################
############################## FFEATURE ENGINEERING ###########################
# Creating new columns/features/targets from existing data
# Preprocess features or Handling Outliers
# more meaningful insights and patterns for machine learning models.
###############################################################################
def calculate_anthropometrics(df, session_column):
    df = calculate_bmi(df, session_column)
    df = calculate_height(df, session_column)
    df = calculate_WHR(df, session_column)

    return df
###############################################################################
# def calculate_behavioral(df, session_column):
#     # Totally 25 fields:
#     # (N=12)
#     df = calculate_cognitive_functioning(df, session_column)
#     # (N=4)
#     df = calculate_depression_score(df, session_column)
#     df = calculate_anxiety_score(df, session_column)
#     df = calculate_cidi_score(df, session_column)
#     df = calculate_neuroticism_score(df, session_column)
#     # (N=6)
#     df = calculate_life_satisfaction(df, session_column)
#     # (N=3)
#     df = calculate_well_being(df, session_column)
    
#     return df
###############################################################################
###############################################################################
def calculate_bmi(df, session_column):
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
    # Assign corresponding session number from the Class:
    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    bmi = session_column.replace(substring_to_remove, "bmi")
    
    df[bmi] = df.apply(lambda row: row[f"21001-{row[session_column]}"], axis=1)

    return df

###############################################################################
def calculate_height(df, session_column):
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
    # Assign corresponding session number from the Class:
    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    substring_to_remove = "session"
    # -----------------------------------------------------------
    height = session_column.replace(substring_to_remove, "height")
    
    df[height] = df.apply(lambda row: row[f"50-{row[session_column]}"], axis=1)

    return df


###############################################################################
def calculate_WHR(df, session_column):
    """Calculate coressponding WHR
    and add "WHR" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: WHR
    """
    # Assign corresponding session number from the Class:
    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    substring_to_remove = "session"
    # -----------------------------------------------------------
    whr = session_column.replace(substring_to_remove, "waist_to_hip_ratio")

    df[whr] = df.apply(lambda row: row[f"48-{row[session_column]}"]/row[f"49-{row[session_column]}"], axis=1)
    
    return df

###############################################################################
def calculate_age(df, session_column):
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
    # Assign corresponding session number from the Class:
    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    substring_to_remove = "session"
    # -----------------------------------------------------------
    age = session_column.replace(substring_to_remove, "age")
    for idx in df.index:
        session = df.loc[idx, session_column]
        if session == 0.0:
            df.loc[idx, age] = df.loc[idx, "Age1stVisit"]
        elif session == 1.0:
            df.loc[idx, age] = df.loc[idx, "AgeRepVisit"]
        elif session == 2.0:
            df.loc[idx, age] = df.loc[idx, "AgeAtScan"]
        elif session == 3.0:
            df.loc[idx, age] = df.loc[idx, "AgeAt2ndScan"]
    
    return df    

###############################################################################
def calculate_days(df, session_column):
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
    # Assign corresponding session number from the Class:

    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    substring_to_remove = "session"
    # -----------------------------------------------------------
    days = session_column.replace(substring_to_remove, "days")
    
    df[days] = df.apply(lambda row: row[f"followup_days-{row[session_column]}"], axis=1)

    return df

###############################################################################
def calculate_years(df, session_column):
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
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    days = session_column.replace(substring_to_remove, "days")
    years = session_column.replace(substring_to_remove, "years")
    
    df.loc[:, years] = df.loc[:, days]/365

    return df

###############################################################################
def calculate_gender(df, session_column):
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
        # Assign corresponding session number from the Class:
            
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(session_column, str), "session_column must be a string!"
        substring_to_remove = "session"
        # -----------------------------------------------------------
        gender = session_column.replace(substring_to_remove, "gender")

        df[gender] = df.apply(lambda row: row[f"31-0.0"], axis=1)
        
        return df
    
###############################################################################
def calculate_handedness(df, session_column, mri_status):
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
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    substring_to_remove = "session"
    # -----------------------------------------------------------
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
    index = df[df.loc[:, session_column].isin([0.0, 1.0, 3.0])].index
    filtered_df = df.loc[index, :]
    idx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 1.0].index
    df.loc[idx, handedness] = 1.0
    idx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 2.0].index
    df.loc[idx, handedness] = 2.0
    idx = filtered_df[filtered_df.loc[:, "1707-0.0"].isin([3.0, -3.0, np.NaN])].index
    df.loc[idx, handedness] = 3.0    
    # ------------------------------------
    if mri_status == "mri":
        index = df[df.loc[:, session_column] == 2.0].index
        filtered_df = df.loc[index, :]
        idx = filtered_df[filtered_df.loc[:, "1707-2.0"] == 1.0].index
        df.loc[idx, handedness] = 1.0
        idx = filtered_df[filtered_df.loc[:, "1707-2.0"] == 2.0].index
        df.loc[idx, handedness] = 2.0
        idx = filtered_df[filtered_df.loc[:, "1707-2.0"].isin([3.0, -3.0, np.NaN])].index
        df_tmp = filtered_df.loc[idx, :]
        idx_tmp = df_tmp[df_tmp.loc[:, "1707-0.0"] == 1.0].index
        df.loc[idx_tmp, handedness] = 1.0
        idx_tmp = df_tmp[df_tmp.loc[:, "1707-0.0"] == 2.0].index
        df.loc[idx_tmp, handedness] = 2.0
        idx_tmp = df_tmp[df_tmp.loc[:, "1707-0.0"].isin([3.0, -3.0, np.NaN])].index
        df.loc[idx_tmp, handedness] = 3.0
        

    return df

###############################################################################
def calculate_neuroticism_score(df, session_column):
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

    
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
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
    
    for idx in df.index:
        session = df.loc[idx, session_column]
        # Add corresponding intences/session that we are looking for 
        # to the list of fields as suffix:
        neuroticism_fields_tmp = [item + f"-{session}" for item in neuroticism_fields]
        # ------------------------------------
        # Add new column "neuroticism_score" by the following process: 
        # Find core_fields answered greather than 0.0
        # And Calculate Sum with min_count parameter=9:
        # df.where is replacing all negative values with NaN
        # min_count=9, means calculate Sum if 
        # at leaset 9 of the fields are answered:
        df.loc[idx, "neuroticism_score"] = \
            df.loc[idx, neuroticism_fields_tmp].where(df.loc[idx, neuroticism_fields_tmp] >= 0.0).sum(axis=0, min_count=9)

    return df

###############################################################################