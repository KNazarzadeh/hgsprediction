
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
    df = calculate_age(df, session_column)    
    df = calculate_gender(df, session_column)
    df = calculate_days(df, session_column)
    df = calculate_years(df, session_column)

    if feature_type == "anthropometrics":
        df = calculate_anthropometrics(df, session_column)
    
    if feature_type == "anthropometrics_age":
        df = calculate_age(df, session_column)
        df = calculate_anthropometrics(df, session_column)
             
    if feature_type == "behavioral":
        df = calculate_behavioral(df, session_column)

    # elif feature_type == "qualification":
    #     df = calculate_qualification(df, session_column)
        
    # elif feature_type == "socioeconomic_status":
    #     df = calculate_socioeconomic_status(df, session_column)
          
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
def calculate_behavioral(df, session_column):
    # Totally 25 fields:
    # (N=12)
    # df = calculate_cognitive_functioning(df, session_column)
    # # (N=4)
    # df = calculate_depression_score(df, session_column)
    # df = calculate_anxiety_score(df, session_column)
    # df = calculate_cidi_score(df, session_column)
    df = calculate_neuroticism_score(df, session_column)
    # # (N=6)
    # df = calculate_life_satisfaction(df, session_column)
    # # (N=3)
    # df = calculate_well_being(df, session_column)
    
    return df
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
        # -----------------------------------------------------------
        df.loc[:, "gender"] = df.loc[:, "31-0.0"]
        
        return df

###############################################################################
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
    substring_to_remove = "session"
    neuroticism_score = session_column.replace(substring_to_remove, "neuroticism_score")
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
        df.loc[idx, neuroticism_score] = df.loc[idx, neuroticism_fields_tmp].where(df.loc[idx, neuroticism_fields_tmp] >= 0.0).sum(axis=0, min_count=9)

    return df

###############################################################################
def calculate_depression_score(df, session_column):
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
    substring_to_remove = "session"
    depression_score = session_column.replace(substring_to_remove, "depression_score")
    # -----------------------------------------------------------    
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
    for idx in df.index:
        session = df.loc[idx, session_column]
        # Add corresponding intences/session that we are looking for 
        # to the list of fields as suffix:
        depression_fields_tmp = [item + f"-{session}" for item in depression_fields]
        # ------------------------------------
        # Add new column "depression_score" by the following process: 
        # Find core_fields answered greather than 0.0
        # And Calculate Sum with min_count parameter=1:
        # df.where is replacing all negative values with NaN
        # min_count=1, means calculate Sum if 
        # at leaset 1 of the fields are answered:
        # df.loc[:, "depression_score"] = df.where(
        #     df[depression_fields] > 0.0).sum(axis=1, min_count=1)
        df.loc[idx, depression_score] = df.loc[idx,depression_fields_tmp].where(df[depression_fields_tmp] > 0.0).sum(axis=1, min_count=1)

    return df

###############################################################################