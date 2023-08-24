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
        motor,
        population,
    ):
        """
        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        motor : str
            Name of the motor which to be analyse.

        population : str
            Name of population to be analyse.

        Returns
        --------
        features_df : list of lists
            List of features.

        """
        self.df = df
        self.motor = motor
        self.population = population

###############################################################################
# Define Motor features which should be extracted from the data.
    def define_motor_features(
        self,
        motor,
    ):
        """Define features.

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
# Define subjectID feature.
    def define_subjectID_features(
        self,
    ):
        """Define features.

        Parameters
        ----------
        None

        Returns
        --------
        baseline_features : list of lists
            List of baseline features.
        """
        subject_id = [
            'eid',   # Subjectid
        ]

        return subject_id

###############################################################################
# Define baseline features from the data.
    def define_baselinecharacteristics_features(
        self,
    ):
        """Define features.

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

###############################################################################
# Define anthropometry features from the data.
    def define_anthropometry_features(
        self,
    ):
        """Define features.

        Parameters
        ----------
        None

        Returns
        --------
        anthropometry_features : list of lists
            List of anthropometry features.
        """
        anthropometry_features = [
            # ====================== Body size measures ======================
            '21001',  # Body mass index (BMI)
            '50',  # Standing height
            'waist_to_hip_ratio',  # Waist to Hip circumference Ratio
        ]

        return anthropometry_features

#############################################################################
# Define earlylife features from the data.
    def define_earlylife_factors_features(
        self,
    ):
        """Define features.

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
    def define_reception_features(
        self,
    ):
        """Define features.

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
# Define Age features At:
# First Visit
# First and Second Scan for MRI subjects.
    def define_age_features(
        self,
    ):
        """Define features.

        Parameters
        ----------
        None

        Returns
        --------
        scan_features : list of lists
            List of scan features.
        """
        age_visit_features = [
            # ====================== Assessment attendance ======================
            'Age1stVisit',  # Age at first Visit the assessment centre
            '21003',  # Age when attended assessment centre
        ]

        age_scan_features = [
            # ====================== Scan attendance ======================
            'AgeAtScan',  # Age at first Scan
            'AgeAt2ndScan',  # Age at second Scan
        ]

        age_features = age_visit_features + age_scan_features

        return age_features

###############################################################################
# Define sociodemographics features which should be extracted from the data.
    def define_sociodemographics_features(
        self,
    ):
        """Define features.

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
    def define_disease_features(
        self,
        population,
    ):
        """Define features.

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
    def define_race_features(
        self,
    ):
        """Define race features.

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
    def define_education_years_features(
        self,
    ):
        """Define education years features.

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
# Compute the extraction of features from the data.
    def extract_features(self):
        """Extract the features from populations data.

        Parameters
        ----------
        None

        Returns
        --------
        features_df : pandas.DataFrame
            DataFrame of data specified.
        """
        motor = self.motor
        df = self.df
        population = self.population

        motor_features = self.define_motor_features(motor)
        baseline_features = self.define_baselinecharacteristics_features()
        sociodemographics_features = self.define_sociodemographics_features()
        reception_features = self.define_reception_features()
        earlylife_features = self.define_earlylife_factors_features()
        anthropometry_features = self.define_anthropometry_features()
        behavioural_features = self.define_behavioural_features()
        # disease_features = self.define_disease_features(population)

        features = \
            motor_features + \
            reception_features + \
            earlylife_features + \
            sociodemographics_features + \
            baseline_features + \
            anthropometry_features + \
            behavioural_features
            # disease_features
        # Extract features from the data
        # get column names that contain the string
        features = [feat + "-" for feat in features]

        subject_id = self.define_subjectID_features()
        age_features = self.define_age_features()
        education_years_features = self.define_education_years_features()
        race_features = self.define_race_features()

        features_df = df.loc[:, subject_id + age_features + education_years_features + race_features]

        for idx, ftrs in enumerate(features):
            ftrs_length = len(ftrs)
            start_char = ftrs[0:ftrs_length]
            df_tmp = df.loc[
                :, df.columns.str.startswith(start_char)]
            features_df = pd.concat(
                [features_df, df_tmp], axis=1).reindex(df_tmp.index)

        return features_df
###############################################################################
# Compute the extraction of features from the data.
    def extract_anthropometrics(self):
        """Extract the anthropometrics features from populations data.
        Parameters
        ----------
        None
        Returns
        --------
        features_df : pandas.DataFrame
            DataFrame of data specified.
        """
        df = self.df

        features = self.define_anthropometry_features()

        df = df[features]

        return df
