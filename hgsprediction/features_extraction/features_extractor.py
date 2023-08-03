#!/usr/bin/env Disorderspredwp3
"""Extract the Features from populations data."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import pandas as pd
# from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# This class extract all required features from data:
class FeaturesExtractor:
    def __init__(
        self,
        df,
        feature_type,
        mri_status,
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
        self.mri_status = mri_status

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
    def extract_gender_features(
        self,
    ):
        """Extract Gender Features.

        Parameters
        ----------
        None

        Returns
        --------
        gender_features : list of lists
            List of gender features.
        """
        gender_features = [
            # ============================ Gender ============================
            '31',
            ]
        return gender_features
###############################################################################    
# Extract anthropometric and age features from the data.
    def extract_age_features(
        self,
    ):
        """Extract Age Features.

        Parameters
        ----------
        None

        Returns
        --------
        age_features : list of lists
            List of age features.
        """
        mri_status = self.mri_status
        
        if mri_status == "nonmri": 
            age_features = [
                # ====================== Assessment attendance ======================
                'Age1stVisit',  # Age at first Visit the assessment centre
                '21003',  # Age when attended assessment centre
            ]
        elif mri_status == "mri": 
            age_features = [
                        # ====================== Scan attendance ======================
                        'AgeAtScan',  # Age at first Scan
                        'AgeAt2ndScan',  # Age at second Scan
                    ]

        return age_features

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
        
        return behavioral_features

###############################################################################
# Define Motor features which should be extracted from the data.
    def extract_motor_features(
        self,
        motor,
    ):
        """Extract motor features.

        Parameters
        ----------
        motor : str
            Name of the motor which to be analyse.

        Returns
        --------
        motor_features : list of lists
            List of motor features.
        """
        # Assign corresponding session number from the Class:
        motor = self.motor
        assert isinstance(motor, str), "motor must be a string!"

        # Available motors:
        available_motor = [
            "handgrip_strength",
            "hgs",
            "handgrip"
        ]
        # Available motors:
        if motor in available_motor:
            motor_features = [
                '46',  # Hand grip strength (left)
                '47',  # Hand grip strength (right)
                'hgs(L+R)',
                'dominant_hgs',
                'nondominant_hgs'
            ]

        return motor_features

###############################################################################
# Define baseline features from the data.
    def extract_baselinecharacteristics_features(
        self,
    ):
        """Extract baseline features.

        Parameters
        ----------
        None

        Returns
        --------
        baseline_features : list of lists
            List of baseline features.
        """
        baseline_features = [
            '31',   # Sex
            '189',  # Townsend deprivation index at recruitment
        ]

        return baseline_features

#############################################################################
# Define earlylife features from the data.
    def extract_earlylife_factors_features(
        self,
    ):
        """Extract earlylife features.

        Parameters
        ----------
        None

        Returns
        --------
        earlylife_features : list of lists
            List of earlylife features.
        """
        earlylife_features = [
            # ====================== Reception ======================
            '1707',  # Handedness
        ]

        return earlylife_features

###############################################################################
# Define Motor features which should be extracted from the data.
    def extract_reception_features(
        self,
    ):
        """Extract reception features.

        Parameters
        ----------
        None

        Returns
        --------
        reception_features : list of lists
            List of reception features.
        """
        reception_features = [
            # ====================== Reception ======================
            '53',     # Date of attending assessment centre
            '21003',  # Age when attended assessment centre
        ]
        return reception_features

###############################################################################
# Define sociodemographics features which should be extracted from the data.
    def extract_sociodemographics_features(
        self,
    ):
        """Extract features.

        Parameters
        ----------
        None

        Returns
        --------
        motor_features : list of lists
            List of sociodemographics features.
        """
        sociodemographics_features = [
            # ====================== Sociodemographics ======================
            '6138',	  # Qualifications
        ]

        return sociodemographics_features

###############################################################################
# Define the features which should be extracted from the data.
# Define the BMC Medicine Paper Features from populations data.
# Paper DOI link: <https://doi.org/10.1186/s12916-022-02490-2>
    def define_behavioural_features(
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
            '1920',  # Mood swings
            '1930',  # Miserableness
            '1940',  # Irritability
            '1950',  # Sensitivity / hurt feelings
            '1960',  # Fed-up feelings
            '1970',  # Nervous feelings
            '1980',  # Worrier / anxious feelings
            '1990',  # Tense / 'highly strung'
            '2000',  # Worry too long after embarrassment
            '2010',  # Suffer from 'nerves'
            '2020',  # Loneliness, isolation
            '2030',  # Guilty feelings
            # '20127', # Neuroticism score on Uk Biobank
            # ------- Depression -------
            '20507',  # Recent feelings of inadequacy
            '20508',  # trouble concentrating on things
            '20510',  # Recent feelings of depression
            '20511',  # Recent poor appetite or overeating
            '20513',  # Recent thoughts of suicide or self-harm
            '20514',  # Recent lack of interest or pleasure in doing things
            '20517',  # Trouble falling or staying asleep, or sleeping too much
            '20518',  # Recent changes in speed/amount of moving or speaking
            '20519',  # Recent feelings of tiredness or low energy
            # ------- Anxiety -------
            '20505',  # Recent easy annoyance or irritability
            '20506',  # Recent feelings or nervousness or anxiety
            '20509',  # Recent inability to stop or control worrying
            '20512',  # Recent feelings of foreboding
            '20515',  # Recent trouble relaxing
            '20516',  # Recent restlessness
            '20520',  # Recent worrying too much about different things'
            # ------- CIDI depression -------
            '20441',  # Ever had prolonged loss of interest in normal activities
            '20446',  # Ever had prolonged feelings of sadness or depression
            '20449',  # Feelings of tiredness during worst episode of depression
            '20450',  # Feelings of worthlessness during worst period of depression
            '20437',  # Thoughts of death during worst depression
            '20435',  # Difficulty concentrating during worst depression
            '20532',  # Did your sleep change?
            '20536',  # Weight change during worst episode of depression
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
# Define disease features which should be extracted from the data.
    def extract_disease_features(
        self,
        population,
    ):
        """Extract features.

        Parameters
        ----------
        population : str
            Name of the population which to be analyse.

        Returns
        --------
        disease_features : list of lists
            List of disease features.
        """
        # Check the disease to choose the features:
        if population == "stroke":
            disease_feature = [
                '42006',  # Date of stroke
                '42007',  # Source of stroke report
                '42008',  # Date of ischaemic stroke
                '42009',  # Source of ischaemic stroke report
                '42010',  # Date of intracerebral haemorrhage
                '42011',  # Source of intracerebral haemorrhage report
                '42012',  # Date of subarachnoid haemorrhage
                '42013',  # Source of subarachnoid haemorrhage report
            ]

        elif population == "parkinson":
            disease_feature = [
                '42030',  # Date of all cause parkinsonism report
                '42031',  # Source of all cause parkinsonism report
                '42032',  # Date of parkinson's disease report
                '42033',  # Source of parkinson's disease report
                '42034',  # Date of progressive supranuclear palsy report
                '42035',  # Source of progressive supranuclear palsy report
                '42036',  # Date of multiple system atrophy report
                '42037',  # Source of multiple system atrophy report
            ]

        return disease_feature

###############################################################################
# Define race feature from the data.
    def extract_race_features(
        self,
    ):
        """Extract race features.

        Parameters
        ----------
        None

        Returns
        --------
        race_features : list of lists
            List of race features.
        """
        race_features = [
            # ====================== Reception ======================
            'Race',
        ]

        return race_features

###############################################################################
# Define education years features from the data.
    def extract_education_years_features(
        self,
    ):
        """Extract education years features.

        Parameters
        ----------
        None

        Returns
        --------
        education_years_features : list of lists
            List of education years features.
        """
        education_years_features = [
            # ====================== Reception ======================
            'YearsOfEducation',  # Number of eductation years
        ]

        return education_years_features

###############################################################################