#!/home/knazarzadeh/miniconda3/envs/disorderspredwp3/bin/python3

"""Extract the Features from populations data.
Based on BMC Paper: https://doi.org/10.1186/s12916-022-02490-2
"""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import pandas as pd
# from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
class ExtractFeatures:
    def __init__(
        self,
        df: pd.DataFrame,
        motor,
        population,
    ):
        """ Etract Features and return the dataframe with extracted features.
        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        motor : str
            Name of the motor which to be analyse.
        population: str
            Name of the population which to  to be analyse
        Return
        ----------
        df : dataframe
            with extracted features
        """
        self.df = df
        self.motor = motor
        self.population = population

###############################################################################
    # Define Motor features which should be extracted from the data.
    def define_motor_features(self):
        """Define Motor features.

        Parameters
        ----------
        None

        Returns
        --------
        motor_features : list of lists
            List of different list of features.
        """
        # Assign corresponding motor type from the Class:
        motor = self.motor
        # Ensuring that the input data is of the expected type
        assert isinstance(motor, str), "motor type must be a string!"

        # The list of acceptable motor types:
        available_motor = [
            "handgrip_strength",
            "hgs",
            "handgrip"
        ]
        # If motor type is accepted then return motor features:
        # Two features are for handgrip strength on UK Biobank
        if motor in available_motor:
            motor_features = [
                '46',  # Hand grip strength (left)
                '47',  # Hand grip strength (right)
            ]

        return motor_features

###############################################################################
    # Define baseline features which should be extracted from the data.
    def define_baselinecharacteristics_features(self):
        """Define Baseline features.

        Parameters
        ----------
        None

        Returns
        --------
        baseline_features : list of lists
            List of different list of features.

        """
        # Requested baseline features on UK Biobank
        # based on BMC paper: https://doi.org/10.1186/s12916-022-02490-2
        baseline_features = [
            '31',   # Sex
            '189',  # Townsend deprivation index at recruitment
        ]

        return baseline_features

###############################################################################
    # Define anthropometry features which should be extracted from the data.
    def define_anthropometry_features(self):
        """Define anthropometry features.

        Parameters
        ----------
        None

        Returns
        --------
        anthropometry_features : list of lists
            List of different list of features.

        """
        # Requested anthropometry features on UK Biobank
        # based on BMC paper: https://doi.org/10.1186/s12916-022-02490-2
        anthropometry_features = [
            # ==================== Body size measures ====================
            '21001',  # Body mass index (BMI)
            '50',  # Standing height
            '48',  # Waist circumference
            '49',  # Hip circumference
        ]

        return anthropometry_features

#############################################################################
    # Define earlylife factors which should be extracted from the data.
    def define_earlylife_factors_features(self):
        """Define earlylife factors features.

        Parameters
        ----------
        None

        Returns
        --------
        earlylife_features : list of lists
            List of different list of features.

        """
        # Requested earlylife features on UK Biobank
        # based on BMC paper: https://doi.org/10.1186/s12916-022-02490-2
        earlylife_features = [
            # ====================== Early Factors ======================
            '1707',  # Handedness
        ]

        return earlylife_features

###############################################################################
    # Define reception features which should be extracted from the data.
    def define_reception_features(self):
        """Define reception features.

        Parameters
        ----------
        None

        Returns
        --------
        reception_features : list of lists
            List of different list of features.

        """
        # Requested reception features on UK Biobank
        # based on BMC paper: https://doi.org/10.1186/s12916-022-02490-2
        reception_features = [
            # ====================== Reception ======================
            '53',     # Date of attending assessment centre
            '21003',  # Age when attended assessment centre
        ]

        return reception_features

###############################################################################
    # Define sociodemographics features that should be extracted from the data
    def define_sociodemographics_features(self):
        """Define sociodemographics features.

        Parameters
        ----------
        None

        Returns
        --------
        sociodemographic_features : list of lists
            List of different list of features.

        """
        # Requested sociodemographics features on UK Biobank
        # based on BMC paper: https://doi.org/10.1186/s12916-022-02490-2
        sociodemographic_features = [
            # ====================== Sociodemographics ======================
            '6138',	  # Qualifications
        ]

        return sociodemographic_features

###############################################################################
    # Define the features which should be extracted from the data.
    # Define the BMC Medicine Paper Features from populations data.
    # Paper DOI link: <https://doi.org/10.1186/s12916-022-02490-2>
    def define_behavioural_features(self):
        """Define behavioural features.

        Parameters
        ----------
        None

        Returns
        --------
        bmc_features : list of lists
            List of different list of features.

        """
        # Requested behavioural features on UK Biobank
        # based on BMC paper: https://doi.org/10.1186/s12916-022-02490-2
        bmc_features = [
            # =================== COGNITIVE FEATURES ===================
            # ------- Fluid intelligence score -------
            # cognitive in clinic
            '20016',  # Fluid intelligence score
            # cognitive online
            # Fluid intelligence score
            # '20191',      # --> BMC paper didn't use this
            # -------  Reaction time task -------
            # cognitive in clinic
            '20023',        # Mean time to correctly identify matches
            # -------  Numeric memory task -------
            # cognitive in clinic
            '4282',         # Maximum digits remembered correctly
            # cognitive online
            # Maximum digits remembered correctly
            # '20240',      # --> BMC paper didn't use this
            # -------  Trail making task -------
            # cognitive in clinic
            # Duration to complete numeric path (trail #1)
            # '6348',       # --> Only available for MRI subjects
            # Duration to complete alphanumeric path (trail #2)
            # '6350',       # --> Only available for MRI subjects
            # cognitive online
            # Duration to complete numeric path (trail #1)
            '20156',        # --> I used this inplace of '6348' field-ID
            # Duration to complete alphanumeric path (trail #2)
            '20157',        # --> I used this inplace of '6350' field-ID
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
            # cognitive in clinic
            '399',          # Number of incorrect matches in round
            '400',          # Time to complete round
            # cognitive online
            # Number of incorrect matches in round
            # '20132',      # --> BMC paper didn't use this
            # Time to complete round
            # '20133',      # --> BMC paper didn't use this
            # ------- Prospective memory result -------
            '20018',        # Prospective memory result
            # ------- symbol digit matches -------
            # cognitive in clinic
            # '23323',      # Number of symbol digit matches attempted
            # '23324',      # Number of symbol digit matches made correctly
            # cognitive online
            # Number of symbol digit matches attempted
            '20195',        # --> Only available for MRI subjects
            # Number of symbol digit matches made correctly
            '20159',        # --> Only available for MRI subjects
            # =================== Depression/Anxiety ===================
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
            # Ever had prolonged loss of interest in normal activities
            '20441',
            # Ever had prolonged feelings of sadness or depression
            '20446',
            # Feelings of tiredness during worst episode of depression
            '20449',
            # Feelings of worthlessness during worst period of depression
            '20450',
            '20437',  # Thoughts of death during worst depression
            '20435',  # Difficulty concentrating during worst depression
            '20532',  # Did your sleep change?
            '20536',  # Weight change during worst episode of depression
            # ==================== Life satisfaction ====================
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

        return bmc_features

###############################################################################
    # Define disease features which should be extracted from the data.
    def define_disease_features(self):
        """Define disease features.

        Parameters
        ----------
        population : str
            Name of the population which to be analyse.

        Returns
        --------
        disease_features : list of lists
            List of different list of features.

        """
        # Assign corresponding population from the Class:
        population = self.population
        # Ensuring that the input data is of the expected type
        assert isinstance(population, str), "population must be a string!"

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
    # Compute the extraction of features from the data.
    def extract_features(self):
        """Extract the features from data.

        Parameters
        ----------
        None

        Returns
        --------
        pandas.DataFrame
            DataFrame of data specified.

        """

        # Assign corresponding motor type from the Class:
        motor = self.motor
        # Assign corresponding data from the Class:
        df = self.df

        # Ensuring that the input data is of the expected type
        assert isinstance(motor, str), "motor type must be a string!"
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"

        # Call and return Motor features:
        motor_features = self.define_motor_features(motor)
        # Call and return baseline features:
        baseline_features = self.define_baselinecharacteristics_features()
        # Call and return sociodemographics features:
        sociodemographics_features = self.define_sociodemographics_features()
        # Call and return reception features:
        reception_features = self.define_reception_features()
        # Call and return earlylife features:
        earlylife_features = self.define_earlylife_factors_features()
        # Call and return anthropometry features:
        anthropometry_features = self.define_anthropometry_features()
        # Call and return behavioural features:
        # based on BMC paper: https://doi.org/10.1186/s12916-022-02490-2
        bmc_features = self.define_behavioural_features()

        # Combine all above features together
        features = \
            motor_features + \
            reception_features + \
            earlylife_features + \
            sociodemographics_features + \
            baseline_features + \
            anthropometry_features + \
            bmc_features

        # Add "-" to end of fields
        features = [feat + "-" for feat in features]
        # Create a sub_df with subject id ('eid')
        sub_df = df['eid']
        # extract each field on dataframe
        # with lenght of each feature and start character
        for idx, ftrs in enumerate(features):
            ftrs_length = len(ftrs)
            start_char = ftrs[0:ftrs_length]
            sub_df_tmp = df.loc[
                :, df.columns.str.startswith(start_char)]
            sub_df = pd.concat(
                [sub_df, sub_df_tmp], axis=1).reindex(sub_df_tmp.index)

        return sub_df
