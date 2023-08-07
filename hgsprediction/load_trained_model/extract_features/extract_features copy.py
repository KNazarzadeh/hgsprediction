#!/usr/bin/env Disorderspredwp3
"""Extract the Features from populations data."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import pandas as pd
# from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# This class extract all required features from data:
class ExtractFeatures:
    def __init__(
        self,
        df,
        feature_type,
    ):
        """
        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        feature_type : str
            Name of feature type to be analyse.

        Returns
        --------
        features : list of lists
            List of features.

        """
        self.df = df
        self.feature_type = feature_type

###############################################################################
    def extract_features(self, feature_type):
            if feature_type == "anthropometrics":
                return self.extract_anthropometric_features()
            elif feature_type == "anthropometrics_gender":
                return self.extract_anthropometric_gender_features()
            elif feature_type == "anthropometrics_age":
                return self.extract_anthropometric_age_features()
            elif feature_type == "behavioral":
                return self.extract_behavioral_features()
            elif feature_type == "behavioral":
                return self.extract_behavioral_gender_features()
            elif feature_type == "anthropometrics_behavioral":
                return self.extract_anthropometrics_behavioral_features()                
            elif feature_type == "anthropometrics_behavioral_gender":
                return self.extract_anthropometrics_behavioral_gender_features()
            else:
                raise ValueError("Invalid feature type. Supported types\n:"
                                    "anthropometrics \n"
                                    "anthropometrics_gender \n"
                                    "anthropometrics_age \n"
                                    "behavioral \n"
                                    "behavioral_gender \n"
                                    "anthropometrics_behavioral \n"
                                    "anthropometrics_behavioral_gender")

###############################################################################
# Extract anthropometric features from the data.
    def extract_anthropometric_features(
        self,
    ):
        """Extract Anthropometrics Features.

        Parameters
        ----------
        None

        Returns
        --------
        anthropometric_features : list of lists
            List of anthropometry features.
        """
        anthropometric_features = [
            # ====================== Body size measures ======================
            '21001',  # Body mass index (BMI)
            '50',  # Standing height
            'waist_to_hip_ratio',  # Waist to Hip circumference Ratio
        ]
        return anthropometric_features
    
###############################################################################    
# Extract anthropometric and gender features from the data.
    def extract_anthropometric_gender_features(
        self,
    ):
        """Extract Anthropometrics and Gender Features.

        Parameters
        ----------
        None

        Returns
        --------
        anthropometric_gender_features : list of lists
            List of anthropometric and gender features.
        """
        anthropometric_gender_features = [
            # ====================== Body size measures ======================
            '21001',  # Body mass index (BMI)
            '50',  # Standing height
            'waist_to_hip_ratio',  # Waist to Hip circumference Ratio
            # ============================ Gender ============================
            '31',
            ]
        return anthropometric_gender_features
###############################################################################    
# Extract anthropometric and age features from the data.
    def extract_anthropometric_age_features(
        self,
    ):
        """Extract Anthropometrics and Age Features.

        Parameters
        ----------
        None

        Returns
        --------
        anthropometric_age_features : list of lists
            List of anthropometric and age features.
        """
        anthropometric_age_features = [
            # ====================== Body size measures ======================
            '21001',  # Body mass index (BMI)
            '50',  # Standing height
            'waist_to_hip_ratio',  # Waist to Hip circumference Ratio
            # ============================= Age ==============================
            '21003',
            ]
        return anthropometric_age_features

###############################################################################
# Define the features which should be extracted from the data.
# Define the BMC Medicine Paper Features from populations data.
# Paper DOI link: <https://doi.org/10.1186/s12916-022-02490-2>
    def extract_behavioral_features(
        self,
    ):
        """Define features.

        Parameters
        ----------
        None

        Returns
        --------
        behavioural_features : list of lists
            List of behavioural features.
        """
        # Define behavioural features based on
        # the BMC Medicine Paper Features from populations data.
        # Paper DOI link: <https://doi.org/10.1186/s12916-022-02490-2>
        behavioural_features = [
            # ====================== COGNITIVE FEATURES ======================
            # ------- Fluid intelligence score -------
            # ---- cognitive in clinic
            '20016',  # Fluid intelligence score
            # ---- cognitive online
            # Fluid intelligence score
            # '20191',      # --> BMC paper didn't use this
            # -------  Reaction time task -------
            # ---- cognitive in clinic
            '20023',        # Mean time to correctly identify matches
            # -------  Numeric memory task -------
            # ---- cognitive in clinic
            '4282',         # Maximum digits remembered correctly
            # ---- cognitive online
            # Maximum digits remembered correctly
            # '20240',      # --> BMC paper didn't use this
            # -------  Trail making task -------
            # ---- cognitive in clinic
            # Duration to complete numeric path (trail #1)
            '6348',       # --> Only available for MRI subjects
            # Duration to complete alphanumeric path (trail #2)
            '6350',       # --> Only available for MRI subjects
            # ---- cognitive online
            # Duration to complete numeric path (trail #1)
            # '20156',        # --> I used this inplace of '20156' field-ID
            # # Duration to complete alphanumeric path (trail #2)
            # '20157',        # --> I used this inplace of '20157' field-ID
            # -------  Matrix pattern completion task -------
            # Number of puzzles correctly solved
            # '6373',       # --> Only available for MRI subjects
            # # -------  Tower rearranging task -------
            # Number of puzzles correct
            # '21004',      # --> Only available for MRI subjects
            # # -------  Paired associate learning task -------
            # Number of word pairs correctly associated
            # '20197',      # --> Only available for MRI subjects
            # -------  Pairs matching task ---------
            # ---- cognitive in clinic
            '399',          # Number of incorrect matches in round
            '400',          # Time to complete round
            # ---- cognitive online
            # Number of incorrect matches in round
            # '20132',      # --> BMC paper didn't use this
            # Time to complete round
            # '20133',      # --> BMC paper didn't use this
            # ------- Prospective memory result -------
            '20018',        # Prospective memory result
            # ------- symbol digit matches -------
            # ---- cognitive in clinic
            '23323',      # Number of symbol digit matches attempted
            '23324',      # Number of symbol digit matches made correctly
            # ---- cognitive online
            # Number of symbol digit matches attempted
            # '20195',        # --> Only available for nonMRI subjects
            # # Number of symbol digit matches made correctly
            # '20159',        # --> Only available for nonMRI subjects
            # ====================== Depression/Anxiety ======================
            # ------- Neuroticism -------
            'neuroticism_score',
            # '1920',  # Mood swings
            # '1930',  # Miserableness
            # '1940',  # Irritability
            # '1950',  # Sensitivity / hurt feelings
            # '1960',  # Fed-up feelings
            # '1970',  # Nervous feelings
            # '1980',  # Worrier / anxious feelings
            # '1990',  # Tense / 'highly strung'
            # '2000',  # Worry too long after embarrassment
            # '2010',  # Suffer from 'nerves'
            # '2020',  # Loneliness, isolation
            # '2030',  # Guilty feelings
            # Extra info: '20127', # Neuroticism score on Uk Biobank
            # ------- Depression -------
            'depression_score',
            # '20507',  # Recent feelings of inadequacy
            # '20508',  # trouble concentrating on things
            # '20510',  # Recent feelings of depression
            # '20511',  # Recent poor appetite or overeating
            # '20513',  # Recent thoughts of suicide or self-harm
            # '20514',  # Recent lack of interest or pleasure in doing things
            # '20517',  # Trouble falling or staying asleep, or sleeping too much
            # '20518',  # Recent changes in speed/amount of moving or speaking
            # '20519',  # Recent feelings of tiredness or low energy
            # ------- Anxiety -------
            'anxiety_score',
            # '20505',  # Recent easy annoyance or irritability
            # '20506',  # Recent feelings or nervousness or anxiety
            # '20509',  # Recent inability to stop or control worrying
            # '20512',  # Recent feelings of foreboding
            # '20515',  # Recent trouble relaxing
            # '20516',  # Recent restlessness
            # '20520',  # Recent worrying too much about different things'
            # ------- CIDI depression -------
            'CIDI_score',
            # '20441',  # Ever had prolonged loss of interest in normal activities
            # '20446',  # Ever had prolonged feelings of sadness or depression
            # '20449',  # Feelings of tiredness during worst episode of depression
            # '20450',  # Feelings of worthlessness during worst period of depression
            # '20437',  # Thoughts of death during worst depression
            # '20435',  # Difficulty concentrating during worst depression
            # '20532',  # Did your sleep change?
            # '20536',  # Weight change during worst episode of depression
            # ===================== Life satisfaction =====================
            '4526',  # Happiness
            '4559',  # Family relationship satisfaction
            '4537',  # Work/job satisfaction
            '4548',  # Health satisfaction
            '4570',  # Friendships satisfaction
            '4581',  # Financial situation satisfaction
            # =================== Subjective well-being ===================
            '20458',  # General happiness
            '20459',  # General happiness with own health
            '20460',  # Belief that own life is meaningful
        ]
        return behavioural_features

###############################################################################
    def extract_behavioral_gender_features(self):
        # Adding Gender to behavioral feature list
        behavioral_gender_features = self.extract_behavioral_features().append('31')
        return behavioral_gender_features

###############################################################################
    def extract_anthropometrics_behavioral_features(self):
        # combine anthropometrics and behavioral features
        anthropometrics_behavioral_features = self.extract_anthropometric_features.extend(self.extract_behavioral_features())
        return anthropometrics_behavioral_features
    
###############################################################################
    def extract_anthropometrics_behavioral_gender_features(self):
        # Adding gender to anthropometrics and behavioral features list
        anthropometrics_behavioral_gender_features = self.extract_anthropometric_features.extend(self.extract_behavioral_features()).append('31')
        return anthropometrics_behavioral_gender_features
    
###############################################################################
