#!/usr/bin/env Disorderspredwp3
"""Define the Features for extracting from populations data."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>
# License: AGPL

import pandas as pd

###############################################################################
# This class extract all required features from data:
class DefineFeatures:
    def __init__(
        self,
        feature_type,
        mri_status,
    ):
        """
        Parameters
        ----------
        feature_type : str

        Returns
        --------
        feature_list : list of lists
            List of features.

        """
        self.feature_type = feature_type
        self.mri_status = mri_status

###############################################################################
# Define the features which should be use for X parameter
# on run_cross_validation.
    def anthropometric_features(self):
        """Define features.

        Parameters
        ----------
        motor : str
            Name of the motor which to be analyse.
        population: str
            Name of the population which to  to be analyse.

        Returns
        --------
        features : list of lists
            List of different list of features.

        """
        anthropometric_features = [
            "bmi",      # '21001' --> BMI
            "height",   #'50' --> Standing Height
            "waist_to_hip_ratio",  # 48/49 --> Waist circumference/ Hip circumference
            ]
        
        anthropometric_age_features = [
        "bmi",      # '21001' --> BMI
        "height",   #'50' --> Standing Height
        "waist_to_hip_ratio",  # 48/49 --> Waist circumference/ Hip circumference
        "age",      # 21003 --> Age when attended assessment centre	
        ]
    
    def behavioral_features(self):
    # Define the features which should be extracted from the data.
    # Define the BMC Medicine Paper Features from populations data.
    # Paper DOI link: <https://doi.org/10.1186/s12916-022-02490-2>    
        if self.mri_status == "nonmri":
            behavioral_features = [
                    # (1) ------- COGNITIVE FEATURES -------
                    # ------- Fluid intelligence score -------
                    # cognitive in clinic
                    '20016',  # Fluid intelligence score
                    # Didn't use cognitive online --> 20191', Fluid intelligence score
                    # (2) ------- Prospective memory result -------
                    '20018',        # Prospective memory result                
                    # (3) -------  Reaction time task -------
                    # cognitive in clinic
                    '20023',  # Mean time to correctly identify matches
                    # (4) -------  Numeric memory task -------
                    # cognitive in clinic
                    '4282',  # Maximum digits remembered correctly
                    # Didn't use cognitive online --> '20240', Maximum digits remembered correctly
                    # (5,6) -------  Trail making task -------
                    # cognitive online --> Only available on non-MRI data
                    '20156',  # Duration to complete numeric path (trail #1)
                    '20157',  # Duration to complete alphanumeric path (trail #2)                
                    # cognitive in clinic --> only available on MRI data 
                        # '6348',  # Duration to complete numeric path (trail #1)
                        # '6350',  # Duration to complete alphanumeric path (trail #2)
                    # (7,8) ------- symbol digit matches -------
                    # cognitive online --> Only available on non-MRI data
                    '20195',  # Number of symbol digit matches attempted
                    '20159',  # Number of symbol digit matches made correctly 
                    # cognitive in clinic --> Only available on MRI data
                        # '23323',  # Number of symbol digit matches attempted
                        # '23324',  # Number of symbol digit matches made correctly
                    # (9,10,11,12) -------  Pairs matching task ---------
                    # The number of attempts as well as the completion time 
                    # required to correctly match 3, 6, 8 pairs of symbol cards 
                    # following a brief visual presentation of these stimuli.                
                    # cognitive in clinic
                    # *** Used only match 3, 6 pairs of symbol cards 
                    # *** (399-1, 399-2, 400-1, 400-2) for non-MRI and MRI.
                    # *** Because, only these tasks are available for non-MRI. 
                    # *** So, ignored other tasks (8 pairs of symbol cards) on MRI data.
                    '399',  # Number of incorrect matches in round
                    '400',  # Time to complete round
                    # Didn't use cognitive online
                        # '20132',  # Number of incorrect matches in round
                        # '20133',  # Time to complete round
                    # (13) ------- Neuroticism -------
                    # Field IDs: [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
                    # Calculation algorithm: The proportion of ‘yes’ responses completed in a 12-item 
                    # Eysenck Personality Questionnaire 
                    # (Participants completing <9 items were excluded from further analysis).
                    'neuroticism_score',
                    # (14) ------- Depression -------
                    # Field IDs: [20507, 20508, 20510, 20511, 20513, 20514, 20517, 20518, 20519]
                    # Calculation algorithm: The sum score from the 9-item 
                    # Patient Health Questionnaire, which was taken during the online follow-up.
                    'depression_score',
                    # (15) ------- Anxiety -------
                    # Field IDs: [20505, 20506, 20509, 20512, 20515, 20516, 20520]
                    # Calculation algorithm: The sum score from the 7-item 
                    # Generalized Anxiety Disorder Questionnaire, 
                    # which was taken during the online follow-up.
                    'anxiety_score',
                    # (16) ------- CIDI depression -------
                    # Field IDs: [20441, 20446, 20449, 20532, 20536, 20435, 20437, 20450]
                    # Calculation algorithm: The depression section of 
                    # the Composite International Diagnostic Interview Short Form (CIDI-SF).
                    'CIDI_score',
                    # (17,18,19) ------- Subjective well-being -------
                    '20458',  # General happiness
                    '20459',  # General happiness with own health
                    '20460',  # Belief that own life is meaningful
                    # ------- PSYCHOSOCIAL FACTORS (MENTAL HEALTH) -------
                    # (20,21,22,23,24,25) ------- Life satisfaction -------
                    '4526',  # Happiness
                    '4559',  # Family relationship satisfaction
                    '4537',  # Work/job satisfaction
                    '4548',  # Health satisfaction
                    '4570',  # Friendships satisfaction
                    '4581',  # Financial situation satisfaction
                    #########################################################
                    ############## Missing Feilds for non-MRI ##############
                    ############# out of 30 behavioral features #############
                    # (26,27) -------  Pairs matching task ---------
                    # *** (399-3, 400-3) Not available for non-MRI and MRI.
                    # *** Because, these tasks aren't available for non-MRI. 
                    # *** So, ignored these tasks on MRI data analysis too.
                    # '399-2',  # Number of incorrect matches in round
                    # '400-2',  # Time to complete round                     
                    # (28) -------  Matrix pattern completion task -------
                    # *** Not available for our non-MRI data
                    # *** So, didn't use for MRI data too
                    # '6373',  # Number of puzzles correctly solved
                    # (29) -------  Tower rearranging task -------
                    # *** Not available for our non-MRI data
                    # *** So, didn't use for MRI data too                
                    # '21004',  # Number of puzzles correct
                    # (30) -------  Paired associate learning task -------
                    # *** Not available for our non-MRI data
                    # *** So, didn't use for MRI data too                
                    # '20197',  # Number of word pairs correctly associated
                ]
        elif self.mri_status == "mri":
            behavioral_features = [
                # (1) ------- COGNITIVE FEATURES -------
                # ------- Fluid intelligence score -------
                # cognitive in clinic
                '20016',  # Fluid intelligence score
                # Didn't use cognitive online --> 20191', Fluid intelligence score
                # (2) ------- Prospective memory result -------
                '20018',        # Prospective memory result                
                # (3) -------  Reaction time task -------
                # cognitive in clinic
                '20023',  # Mean time to correctly identify matches
                # (4) -------  Numeric memory task -------
                # cognitive in clinic
                '4282',  # Maximum digits remembered correctly
                # Didn't use cognitive online --> '20240', Maximum digits remembered correctly
                # (5,6) -------  Trail making task -------
                # cognitive in clinic --> only available on MRI data 
                '6348',  # Duration to complete numeric path (trail #1)
                '6350',  # Duration to complete alphanumeric path (trail #2)            
                # cognitive online --> Only available on non-MRI data
                    # '20156',  # Duration to complete numeric path (trail #1)
                    # '20157',  # Duration to complete alphanumeric path (trail #2)
                # (7,8) ------- symbol digit matches -------
                # cognitive in clinic --> Only available on MRI data
                '23323',  # Number of symbol digit matches attempted
                '23324',  # Number of symbol digit matches made correctly
                # cognitive online --> Only available on non-MRI data
                    # '20195',  # Number of symbol digit matches attempted
                    # '20159',  # Number of symbol digit matches made correctly                 
                # (9,10,11,12) -------  Pairs matching task ---------
                # The number of attempts as well as the completion time 
                # required to correctly match 3, 6, 8 pairs of symbol cards 
                # following a brief visual presentation of these stimuli.                
                # cognitive in clinic
                # *** Used only match 3, 6 pairs of symbol cards 
                # *** (399-1, 399-2, 400-1, 400-2) for non-MRI and MRI.
                # *** Because, only these tasks are available for non-MRI. 
                # *** So, ignored other tasks (8 pairs of symbol cards) on MRI data.
                '399',  # Number of incorrect matches in round
                '400',  # Time to complete round
                # Didn't use cognitive online
                    # '20132',  # Number of incorrect matches in round
                    # '20133',  # Time to complete round
                # (13) ------- Neuroticism -------
                # Field IDs: [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
                # Calculation algorithm: The proportion of ‘yes’ responses completed in a 12-item 
                # Eysenck Personality Questionnaire 
                # (Participants completing <9 items were excluded from further analysis).
                'neuroticism_score',
                # (14) ------- Depression -------
                # Field IDs: [20507, 20508, 20510, 20511, 20513, 20514, 20517, 20518, 20519]
                # Calculation algorithm: The sum score from the 9-item 
                # Patient Health Questionnaire, which was taken during the online follow-up.
                'depression_score',            
                # (15) ------- Anxiety -------
                # Field IDs: [20505, 20506, 20509, 20512, 20515, 20516, 20520]
                # Calculation algorithm: The sum score from the 7-item 
                # Generalized Anxiety Disorder Questionnaire, 
                # which was taken during the online follow-up.
                'anxiety_score',
                # (16) ------- CIDI depression -------
                # Field IDs: [20441, 20446, 20449, 20532, 20536, 20435, 20437, 20450]
                # Calculation algorithm: The depression section of 
                # the Composite International Diagnostic Interview Short Form (CIDI-SF).
                'CIDI_score',
                # (17,18,19) ------- Subjective well-being -------
                '20458',  # General happiness
                '20459',  # General happiness with own health
                '20460',  # Belief that own life is meaningful
                # ------- PSYCHOSOCIAL FACTORS (MENTAL HEALTH) -------
                # (20,21,22,23,24,25) ------- Life satisfaction -------
                '4526',  # Happiness
                '4559',  # Family relationship satisfaction
                '4537',  # Work/job satisfaction
                '4548',  # Health satisfaction
                '4570',  # Friendships satisfaction
                '4581',  # Financial situation satisfaction
                #########################################################
                ############## Missing Feilds for non-MRI ##############
                ############# out of 30 behavioral features #############
                # (26,27) -------  Pairs matching task ---------
                # *** (399-3, 400-3) Not available for non-MRI and MRI.
                # *** Because, these tasks aren't available for non-MRI. 
                # *** So, ignored these tasks on MRI data analysis too.
                # '399-2',  # Number of incorrect matches in round
                # '400-2',  # Time to complete round                     
                # (28) -------  Matrix pattern completion task -------
                # *** Not available for our non-MRI data
                # *** So, didn't use for MRI data too
                # '6373',  # Number of puzzles correctly solved
                # (29) -------  Tower rearranging task -------
                # *** Not available for our non-MRI data
                # *** So, didn't use for MRI data too                
                # '21004',  # Number of puzzles correct
                # (30) -------  Paired associate learning task -------
                # *** Not available for our non-MRI data
                # *** So, didn't use for MRI data too                
                # '20197',  # Number of word pairs correctly associated
            ]


    if feature_type == "bodysize":
        X = dataframe.filter(regex='^('+'|'.join(bodysize_features)+')', axis=1).columns.to_list()
        
    elif feature_type == "bodysize+age":
        bodysize_age = bodysize_features + ['Age1stVisit']
        X = dataframe.filter(regex='^('+'|'.join(bodysize_age)+')', axis=1).columns.to_list()
        
    elif feature_type == "bodysize+gender":
        bodysize_gender = bodysize_features + ['31']
        X = dataframe.filter(regex='^('+'|'.join(bodysize_gender)+')', axis=1).columns.to_list()
        
    elif feature_type == "cognitive":
        X = dataframe.filter(regex='^('+'|'.join(cognitive_features)+')', axis=1).columns.to_list()
        
    elif feature_type == "cognitive+gender":
        cognitive_gender = cognitive_features + ['31']
        X = dataframe.filter(regex='^('+'|'.join(cognitive_gender)+')', axis=1).columns.to_list()
        
    elif feature_type == "bodysize+cognitive":
        bodysize_cognitive = bodysize_features + cognitive_features
        X = dataframe.filter(regex='^('+'|'.join(bodysize_cognitive)+')', axis=1).columns.to_list()

    elif feature_type == "bodysize+cognitive+gender":
        bodysize_cognitive = bodysize_features + cognitive_features
        bodysize_cognitive_gender = bodysize_cognitive + ['31']
        X = dataframe.filter(regex='^('+'|'.join(bodysize_cognitive_gender)+')', axis=1).columns.to_list()


    return X
