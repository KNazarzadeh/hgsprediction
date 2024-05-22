# Authors: Kimia Nazarzadeh <kimia.nazarzadeh@uk-koeln.de>

###############################################################################
# Define Class UkbbParams
# to get different parameters on different functions for:
# - healthy/stroke/parkinson populations
# - motor performance marker/type
# - Othe parameters based on UK Biobank field-IDs/Category-IDs
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
class UkbbParams:
    def __init__(
        self,
        motor_type,
        ishealthy,
        population_name,
        mri,
    ):
        """
        Parameters
        ----------
        motor_type : str
            The type of Motor that desired to analysis
        ishealthy : binary
            The binary values for healthy and stroke/parkinson populations:
            1 --> healthy
            0 --> stroke or parkinson
        population_name : str
            The str values for different types of disease or healthy controls:
            healthy --> Healthy controls
                    excluded: Specific Criteria/disease
            stroke  --> Stroke --> All stroke subjects -
                   excluded: Specific Criteria &
                   included: 'I63'& 'I61' stroke types
            parkinson  --> parkinson --> All parkinson subjects -
                   excluded: Specific Criteria &
                   included: 'G20'
        mri : binary
            The binary values of MRI data status for subjects with/without MRI
            1 --> MRI data (subjects with MRI data only)
            0 --> non_MRI data (all subjects without concern MRI data)
        """
        self.motor_type = motor_type
        self.ishealthy = ishealthy
        self.population_name = population_name
        self.mri = mri

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Healthy, Stroke and Parkinson populations
# Based on ICD10 (International Classification disease) codes
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_healthy_params(
        self,
    ):
        """
        Get the list of criteria/disease based on ICD10 codes
        for  and --incon and --excon flags on ukbb_parser command.
            -----------
            Parameters
            -----------
            No inputs
                all parameters and function are using from class UkbbParams
            -----------
            Returns
            -----------
            icd10_excon : list of str
                List of diseases that must exclude
            icd10_incon : list of str
                List of diseases that must include
        """
        icd10_excon = [
            'I60-I69',    # Cerebrovascular diseases
            'F',          # V - Mental and behavioural disorders
            'G',          # VI - Diseases of the nervous system
            'M',          # XIII - Diseases of the musculoskeletal system
                          # and connective tissue
            'S',          # XIX - Injury, poisoning and certain other
                          # consequences of external causes
            ] 
        icd10_incon = []
        
        return  icd10_excon, \
                icd10_incon

###############################################################################
    def get_stroke_params(
        self,
    ):
        """ 
        Get the ukbb string command of Stroke population to be included to
        the ukbb_parser command.
            -----------
            Parameters
            -----------
            No inputs
                all parameters and function are using from class UkbbParams
            -----------
            Return
            -----------
            icd10_excon : list of str
                List of diseases that must exclude
            icd10_incon : list of str
                List of diseases that must include
        -----------------------------------------------------------------------
        The int values for different types of stroke:
        Based on ICD10 --> overal stroke - 
            excluded: [F, Some of G, M, S] &
            included:'I63' and 'I61'stroke types &
            ignored: 'I64', # IX - Undetermined type
                    'I60', # IX - Subarachnoid Haemorrhage (SAH)
                    'I62', # IX - Other nontraumatic intracranial 
                                    haemorrhage
                    'G40-G47', # VI - Episodic and paroxysmal disorders
                    'G50-G53', # VI - Nerve, nerve root and plexus disorders
                                    G50 - Disorders of trigeminal nerve
                                    G51 - Facial nerve disorders
                                    G52 - Disorders of other cranial nerves
                                    G53 - Cranial nerve disorders in diseases classified elsewhere
                    'G62-G64', # VI - Polyneuropathies and other disorders of the peripheral nervous system
                                    G62 - Other polyneuropathies
                                    G63 - Polyneuropathy in diseases classified elsewhere
                                    G64 - Other disorders of peripheral nervous system
        -----------------------------------------------------------------------
        """    
        icd10_excon = [
            'G00-G09',  # VI - Inflammatory diseases of the central nervous system
            'G10-G14',  # VI - Systemic atrophies primarily affecting the central nervous system
            'G20-G26',  # VI - Extrapyramidal and movement disorders
            'G30-G32',  # VI - Other degenerative diseases of the nervous system
            'G35-G37',  # VI - Demyelinating diseases of the central nervous system
            'G54-G59',  # VI - Nerve, nerve root and plexus disorders
            'G60-G61',  # VI - Polyneuropathies and other disorders of the peripheral nervous system
                                # G60 - Hereditary motor and sensory neuropathy
                                # G61 - Inflammatory polyneuropathy
            'G70-G73',  # VI - Diseases of myoneural junction and muscle
            'G80-G83',  # VI - Cerebral palsy and other paralytic syndromes
            'G90-G99',  # VI - Other disorders of the nervous system
            
            'F',        # V - Mental and behavioural disorders   
            'M',        # XIII - Diseases of the musculoskeletal 
                        # system and connective tissue
            'S',        # XIX - Injury, poisoning and certain other 
                        # consequences of external causes
            ]
        icd10_incon = [
            'I63',      # IX - Ischaemic stroke
            'I61',      # IX - Intracerebral Haemorrhage (ICH)
            ]

        return  icd10_excon, \
                icd10_incon

# -----------------------------------------------------------------------------#
    def get_parkinson_params(
        self,
    ):
        """ 
        Get the ukbb string command of Parkinson's disease population to be included to
        the ukbb_parser command.
            -----------
            Parameters
            -----------
            No inputs
                all parameters and function are using from class UkbbParams
            -----------
            Returns
            -----------
            icd10_excon : list of str
                List of diseases that must exclude
            icd10_incon : list of str
                List of diseases that must include
        -----------------------------------------------------------------------
        The int values for different types of parkinson:
        Based on ICD10 --> overal parkinson - 
            excluded:[F, G, M, S] &
            included:'G20' Parkinson's disease
            ignored:'G40-G47', # VI - Episodic and paroxysmal disorders
                    'G50-G53', # VI - Nerve, nerve root and plexus disorders
                        G50 - Disorders of trigeminal nerve
                        G51 - Facial nerve disorders
                        G52 - Disorders of other cranial nerves
                        G53 - Cranial nerve disorders in diseases classified elsewhere
                    'G62-G64', # VI - Polyneuropathies and other disorders of the peripheral nervous system
                        G62 - Other polyneuropathies
                        G63 - Polyneuropathy in diseases classified elsewhere
                        G64 - Other disorders of peripheral nervous system
        -----------------------------------------------------------------------
        """    
        icd10_excon = [
            'G00-G09',  # VI - Inflammatory diseases of the central nervous system
            'G10-G14',  # VI - Systemic atrophies primarily affecting the central nervous system
            'G21-G26',
            'G30-G32',  # VI - Other degenerative diseases of the nervous system
            'G35-G37',  # VI - Demyelinating diseases of the central nervous system
            'G54-G59',  # VI - Nerve, nerve root and plexus disorders
            'G60-G61',  # VI - Polyneuropathies and other disorders of the peripheral nervous system
                                # G60 - Hereditary motor and sensory neuropathy
                                # G61 - Inflammatory polyneuropathy
            'G70-G73',  # VI - Diseases of myoneural junction and muscle
            'G80-G83',  # VI - Cerebral palsy and other paralytic syndromes
            'G91-G99',  # VI - Other disorders of the nervous system
            
            'F',        # V - Mental and behavioural disorders   
            'M',        # XIII - Diseases of the musculoskeletal 
                        # system and connective tissue
            'S',        # XIX - Injury, poisoning and certain other 
                        # consequences of external causes
            'I60-I69',  # IX - Cerebrovascular Diseases
            ]
        icd10_incon = [
            # 'G20-G26',      # VI - Idiopathic Parkinsons Disease
            'G20',  # VI - Parkinson disease   
            ]

        return  icd10_excon, \
                icd10_incon
                
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Healthy, Stroke and Parkinson populations
# Based on ICD10 (International Classification disease) codes
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_depression_params(
        self,
    ):
        """
        Get the list of criteria/disease based on ICD10 codes
        for  and --incon and --excon flags on ukbb_parser command.
            -----------
            Parameters
            -----------
            No inputs
                all parameters and function are using from class UkbbParams
            -----------
            Returns
            -----------
            icd10_excon : list of str
                List of diseases that must exclude
            icd10_incon : list of str
                List of diseases that must include
        """
        icd10_excon = [
            'F00-F09',    # V -  Organic, including symptomatic, mental disorders
            'F10-F19',    # V -  Mental and behavioural disorders due to psychoactive substance use
            'F20-F29',    # V -  Schizophrenia, schizotypal and delusional disorders
            'F30-F31',    # V -  Mood [affective] disorders:
                                # F30:Manic episode
                                # F31:Bipolar affective disorder
            'F34-F39',    # V -  Organic, including symptomatic, mental disorders:
                                # F34:Persistent mood [affective] disorders
                                # F38:Other mood [affective] disorders
                                # F39:Unspecified mood [affective] disorder
            'F40-F48',    # V -  Neurotic, stress-related and somatoform disorders
            'F50-F59',    # V -  Behavioural syndromes associated with physiological disturbances and physical factors
            'F60-F69',    # V -  Disorders of adult personality and behaviour
            'F70-F79',    # V -  Mental retardation
            'F80-F89',    # V -  Disorders of psychological development
            'F90-F98',    # V -  Behavioural and emotional disorders with onset usually occurring in childhood and adolescence            
            'F99',        # V -  Unspecified mental disorder            
            
            'G',          # VI - Diseases of the nervous system
            'M',          # XIII - Diseases of the musculoskeletal system
                          # and connective tissue
            'S',          # XIX - Injury, poisoning and certain other
                          # consequences of external causes
            'I60-I69',    # Cerebrovascular diseases                          
            ] 
        icd10_incon = [
            'F32-F33',      # V - F32:Depressive episode & F33:Recurrent depressive disorder
        ]
        
        return  icd10_excon, \
                icd10_incon

###############################################################################
# --------------------------------#
# Define motor function 
# to get the specific motor filed-IDs/ Categor-IDs Based on UK Biobank
# --------------------------------#
    def get_motor_params(
        self,
    ):
        """ 
        Get the list of Motor Performance Field IDs to be included by --inhdr 
        on ukbb_parser command for different Motor Performance Analysis
            -----------
            Parameters
            -----------
            analysis_motor : str
                The desired Motor Performance type to be analysed
                The parameters input from the class paramater
            -----------
            Returns
            -----------
            motor_inhdr : list of str
                The list of Motor Performance primary and related Field IDs 
                to be included to ukbb_parser command
        """
        # ------ Handgrip Strength (HGS) as the Motor Performance
        if self.motor_type is None:
            raise ValueError(
                "Please define the motor type for analysis!"
            )
        # if self.motor_type.isin(["handgrip_strength", "hgs", "HGS"]):
        if self.motor_type in ["hgs", "handgrip_strength"]:
            # 46 ---- for Left Handgrip Strength
            # 47 ---- for Right Handgrip Strength
            motor_incat = [
                # '100019',   # Hand grip strength (Physical Measures - Category 100006)
            ]
            motor_inhdr = [
                '46',       # for Left Handgrip Strength
                '47',       # for Right Handgrip Strength
                '1707',     # Handedness (chirality/laterality)--
                            # -- Category 100033 (Early life factors) 
                '20044',	# Reason for skipping grip strength (left) --
                            # -- Category 100019 Hand grip strength
                '20043',	# Reason for skipping grip strength (right) --
                            # -- Category 100019 Hand grip strength
                # '38', 	# Hand grip dynamometer device ID --
                #           # -- Category 100019 Hand grip strength
            ]
            motor_exhdr = []
            motor_excat = []

        return motor_incat, \
               motor_inhdr, \
               motor_exhdr, \
               motor_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define assessment centre parameters 
# to get the specific assessment filed-IDs/ Categor-IDs Based on UK Biobank
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_assessment_params(
        self,
    ):
        """ 
        Get the list of Assessment Field IDs to be included by --inhdr 
        on ukbb_parser command
            -----------
            Parameters
            -----------
            No inputs
                all parameters and function are using from class UkbbParams
            -----------
            Returns
            -----------
            assessment_inhdr : list of str
                The list of Assessment primary and related Field IDs 
                to be included to ukbb_parser command
        """
        assessment_incat = []
        assessment_inhdr = [
            '54',	    # UK Biobank assessment centre
            '53',	    # Date of attending assessment centre
            '55',	    # Month of attending assessment centre
            '21003',    # Age when attended assessment centre
        ]
        assessment_exhdr = []
        assessment_excat = []

        return assessment_incat, \
               assessment_inhdr, \
               assessment_exhdr, \
               assessment_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define stroke outcome parameters on UK Biobank 
# --------------------------------#
# ---------------------------------------------------------------------------- #
    def get_stroke_outcomes(self):
        """ 
        Get the list of stroke outcome Field, Category IDs to be included/excluded
        by --inhdr, --incat, --excat 
        on ukbb_parser command.
            -----------
            Parameters
            -----------
            No inputs
            all parameters and function are using from class UkbbParams.
            -----------
            Returns
            -----------
            stroke_outcome_inhdr: list of str
                The list of stroke outcome Field IDs to be included by --inhdr
                to ukbb_parser command

            stroke_outcome_exhdr: list of str
            stroke_outcome_incat: list of str
            stroke_outcome_excat: list of str
        """
        stroke_outcome_incat = [
            '43', # Stroke outcomes
        ]
        stroke_outcome_inhdr = [
            # # Category 2409 - Circulatory system disorders-First occurrences
            # '131366',   # Date I63 first reported (cerebral infarction)
            # '131362',   # Date I61 first reported (intracerebral haemorrhage)
            # '131360',   # Date I60 first reported (subarachnoid haemorrhage)
        ]
        stroke_outcome_exhdr = []
        stroke_outcome_excat = []

        return stroke_outcome_incat, \
               stroke_outcome_inhdr, \
               stroke_outcome_exhdr, \
               stroke_outcome_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define parkinson's disease outcome parameters on UK Biobank 
# --------------------------------#
# ---------------------------------------------------------------------------- #
    def get_parkinson_outcomes(self):
        """ 
        Get the list of parkinson outcome Field, Category IDs to be included/excluded
        by --inhdr, --incat, --excat 
        on ukbb_parser command.
            -----------
            Parameters
            -----------
            No inputs
            all parameters and function are using from class UkbbParams.
            -----------
            Returns
            -----------
            PD_outcome_inhdr: list of str
                The list of parkinson outcome Field IDs to be included by --inhdr
                to ukbb_parser command

            PD_outcome_exhdr: list of str
            PD_outcome_incat: list of str
            PD_outcome_excat: list of str
        """
        pd_outcome_incat = [
            '50', # parkinson outcomes
        ]
        pd_outcome_inhdr = [
            # 42030	Date of all cause parkinsonism report
            # 42031	Source of all cause parkinsonism report
            # 42032	Date of parkinson's disease report
            # 42033	Source of parkinson's disease report
            # 42034	Date of progressive supranuclear palsy report
            # 42035	Source of progressive supranuclear palsy report
            # 42036	Date of multiple system atrophy report
            # 42037	Source of multiple system atrophy report
        ]
        pd_outcome_exhdr = []
        pd_outcome_excat = []

        return pd_outcome_incat, \
               pd_outcome_inhdr, \
               pd_outcome_exhdr, \
               pd_outcome_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define parkinson's disease outcome parameters on UK Biobank 
# --------------------------------#
# ---------------------------------------------------------------------------- #
    def get_depression_outcomes(self):
        """ 
        Get the list of parkinson outcome Field, Category IDs to be included/excluded
        by --inhdr, --incat, --excat 
        on ukbb_parser command.
            -----------
            Parameters
            -----------
            No inputs
            all parameters and function are using from class UkbbParams.
            -----------
            Returns
            -----------
            depression_outcome_inhdr: list of str
                The list of parkinson outcome Field IDs to be included by --inhdr
                to ukbb_parser command

            depression_outcome_exhdr: list of str
            depression_outcome_incat: list of str
            depression_outcome_excat: list of str
        """
        depression_outcome_incat = [
            '138', # Online follow-up ⏵ Mental health ⏵ Depression 
            '1502' # Online follow-up ⏵ Mental well-being ⏵ Depression 
        ]
        depression_outcome_inhdr = []
        depression_outcome_exhdr = []
        depression_outcome_excat = []

        return depression_outcome_incat, \
               depression_outcome_inhdr, \
               depression_outcome_exhdr, \
               depression_outcome_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define demographic parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_demographic_params(
        self,
    ):
        """ 
        Get the Feild IDs of Demographic features for the ukbb_parser command.            
            -----------
            Parameters
            -----------
            No inputs
                all parameters and function are using from class UkbbParams
            -----------
            Returns
            -----------
            data_field_list : list of str
                The ukb string command to be included to ukbb_parser command
        """
        demographic_incat = []  
        demographic_inhdr = [
            # '21003',	# Age when attended assessment centre --> Reception
            # '54',	    # UK Biobank assessment centre --> Reception  
            # '53',	    # Date of attending assessment centre --> Reception  
            '21000',	# Ethnic background	Ethnicity --> Reception
        ]
        demographic_exhdr = []
        demographic_excat =[]
    
        return demographic_incat, \
               demographic_inhdr, \
               demographic_exhdr, \
               demographic_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Sociodemographic parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_sociodemographics_param(
        self,
    ):
        socio_incat = [
            # '100064',   # Employment (Sociodemographics-Category 100062 <- Touchscreen)
            # '100073',   # Employment (Verbal interview - Category 100071)
            # '100065',   # Ethnicity (Sociodemographics-Category 100062 <- Touchscreen)
            # '100063',   # Education (Sociodemographics-Category 100062 <- Touchscreen)
        ]
        socio_inhdr = [
            '816',	    # Job involves heavy manual or physical work --> Employment  
            # '806',	Job involves mainly walking or standing	Employment  
            '6138',	# Qualifications
            '10722',	# Qualifications (pilot)
            '845',	# Age completed full time education
        ]
        socio_exhdr = []
        socio_excat = []

        return socio_incat, \
               socio_inhdr, \
               socio_exhdr, \
               socio_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Lifestyle parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_lifestyle_params(
        self,
    ):
        lifestyle_incat = [
            '100054',   # Physical activity
            '100053',   # Electronic device use
            '100057',   # Sleep
            '100058',   # Smoking
            '100052',   # Diet
            '100051',   # Alcohol
            '100056',   # Sexual factors
        ]
        lifestyle_inhdr = []
        lifestyle_exhdr = []
        lifestyle_excat = []

        return lifestyle_incat, \
               lifestyle_inhdr, \
               lifestyle_exhdr, \
               lifestyle_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Anthropometry parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_body_measures_params(
        self,
    ):
        # ---------- Body size measures - Category 100010:
        body_size_incat = [
            # '100010',   # Body size measures - Category 100010
        ]
        # Parent Category: Anthropometry -> 100008
        # Physical measures -> Assessment Centre
        # Contains h following Field IDs (inhdrs):
        body_size_inhdr = [
            '50',	    # Standing height
            '51',       # Seated height
            '20015',	# Sitting height
            '3077',	    # Seating box height
            '20015',    # calculated as the difference between Field 51 and Field 3077
            # '20048',	# Reason for skipping sitting height
            # '20047',	# Reason for skipping standing height
            # '20041',	# Reason for skipping weight
            '12144',	# Height
            '21002',    # Weight
            # '12143',	# Weight (pre-imaging)
                        # Weight measured prior to imaging stages. 
                        # Required by DXA device for calibration. 
            '21001',	# Body mass index (BMI)
            '3160',	    # Weight (manual entry)

            # '21',       # Weight method
            # '39',       # Height measure device ID
            # '40',       # Manual scales device ID
            # '41',       # Seating box device ID
            # '44',       # Tape measure device ID
            '48',       # Waist circumference
            '49',       # Hip circumference
            # '20046',    # Reason for skipping hip measurement
        ]
        body_size_exhdr = []
        body_size_excat = []

        # ---------- Body composition by impedance - Category 100009:
        body_size_impedance_incat = []
        # '100009', # Body composition by impedance - Category 100009
        #             # Parent Category: Anthropometry -> 
        #             # Physical measures -> Assessment Centre
        #             # Contains h following Field IDs (inhdrs):
        body_size_impedance_inhdr = [
            # '23113',	# Leg fat-free mass (right)
            # '23118',	# Leg predicted mass (left)
            # '23114',	# Leg predicted mass (right)
            # '23123',	# Arm fat percentage (left)
            # '23119',	# Arm fat percentage (right)
            # '23124',	# Arm fat mass (left)
            # '23120',	# Arm fat mass (right)
            # '23121',	# Arm fat-free mass (right)
            # '23125',	# Arm fat-free mass (left)
            # '23126',	# Arm predicted mass (left)
            # '23122',	# Arm predicted mass (right)
            # '23127',	# Trunk fat percentage
            # '23128',	# Trunk fat mass
            # '23129',  # Trunk fat-free mass
            # '23130',	# Trunk predicted mass
            # '23105',	# Basal metabolic rate
            # '23099',	# Body fat percentage
            # '23100',	# Whole body fat mass
            # '23101',	# Whole body fat-free mass
            # '23102',	# Whole body water mass
            # '23115',	# Leg fat percentage (left)
            # '23111',	# Leg fat percentage (right)
            # '23116',	# Leg fat mass (left)
            # '23112',	# Leg fat mass (right)
            # '23117',	# Leg fat-free mass (left)
            # '23106',	# Impedance of whole body
            # '23110',	# Impedance of arm (left)
            # '23109',	# Impedance of arm (right)
            # '23108',	# Impedance of leg (left)
            # '23107',	# Impedance of leg (right)
            # '6218',	# Impedance of whole body, manual entry
            # '6222',	# Impedance of arm, manual entry (left)
            # '6221',	# Impedance of arm, manual entry (right)
            # '6220',	# Impedance of leg, manual entry (left)
            # '6219',	# Impedance of leg, manual entry (right)
        ]
        body_size_impedance_exhdr = []
        body_size_impedance_excat = []

        return body_size_incat, \
               body_size_inhdr, \
               body_size_exhdr, \
               body_size_excat, \
               body_size_impedance_incat, \
               body_size_impedance_inhdr, \
               body_size_impedance_exhdr, \
               body_size_impedance_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Cognitive at Clinic parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_cognitive_clinic_params(
        self,
    ):
        # Based on BMC Paper: Rongtao Jiang, 2022
        cognitive_incat = [
            '100032',       # Reaction time task
            '100029',       # Numeric memory task
            '100027',       # Fluid intelligence/reasoning task
            '505',          # Trail making task
            '501',          # Matrix pattern completion task
            '503',          # Tower rearranging task
            '502',          # Symbol digit substitution task
            '506',          # Paired associate learning task
            '100031',       # Prospective memory task
            '100030',       # Pairs matching task
            # '504',          # Picture vocabulary task
            '100028',       # Lights pattern memory task
            '100077',       # Word production task
        ]
        cognitive_inhdr = [
            # ------------ Fluid intelligence score --------------------
            # '20016',    # Fluid intelligence score
            # # '20128',    # Number of fluid intelligence questions attempted within time limit
            # # # # -------  Reaction time task ---------
            # '20023',    # Reaction time 
            # # # # -------  Numeric memory task ---------
            # '4282',     # Numeric memory
            # # # # -------  Trail making task ---------
            # '6348',     # Duration to complete numeric path (trail #1)	  
            # '6350',     # Duration to complete alphanumeric path (trail #2)  
            # # # # -------  Matrix pattern completion task ---------
            # '6373',     # Number of puzzles correctly solved
            # # # # -------  Tower rearranging task ---------
            # '21004',    # Number of puzzles correct
            # # # # -------  Paired associate learning task ---------
            # '20197',    # Number of word pairs correctly associated
            # # # # -------  Pairs matching task ---------
            # '399',      # Number of incorrect matches in round
            # '400',      # Time to complete round
            # # ---------- Prospective memory result ---------------------
            # '20018',        # Prospective memory result
            # # # --------- symbol digit matches -----------------------
            # '23323',	# Number of symbol digit matches attempted
            # '23324',	# Number of symbol digit matches made correctly
        ]
        cognitive_exhdr = []
        cognitive_excat = []

        return cognitive_incat, \
               cognitive_inhdr, \
               cognitive_exhdr, \
               cognitive_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Cognitive Online parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_cognitive_online_params(
        self,
    ):
        cognitive_online_incat = [
            # '155',          # Mood
            # '118',          # Fluid intelligence/reasoning task
            # '121',          # Trail making task
            # '122',          # Symbol digit substitution task
            # '117',          # Pairs matching task
            # '120',          # Numeric memory task
        ]
        cognitive_online_inhdr = [
            '23045',    # Very nervous mood over last week
            '23046',    # Down in dumps over last week
            '23047',    # Felt calm over last week
            '23072',    # Downhearted and depressed over last week
            '23076',    # Happy over last week
            '23079',    # When mood described
            # ------------ Trail making task --------------------
            '20156',	# Duration to complete numeric path (trail #1)
            '20157',	# Duration to complete alphanumeric path (trail #2)
            '20247',	# Total errors traversing numeric path (trail #1)
            '20248',	# Total errors traversing alphanumeric path (trail #2)
            '20149',	# Interval between previous point and current one in numeric path (trail #1)
            '20155',	# Interval between previous point and current one in alphanumeric path (trail #2)
            '20147',	# Errors before selecting correct item in numeric path (trail #1)
            '20148',	# Errors before selecting correct item in alphanumeric path (trail #2)
            '20246',	# Trail making completion status
            # '20136',	# When trail making test completed xxxxxxxxx
            # ------------ Fluid intelligence score --------------------
            '20191',    # Fluid intelligence score
            # '20135',    # When fluid intelligence test completed xxxxxxxxx
            # '20165',    # FI1 : numeric addition test xxxxxxxxx
            # '20167',    # FI2 : identify largest number xxxxxxxxx
            # '20169',    # FI3 : word interpolation xxxxxxxxx
            # '20171',    # FI4 : positional arithmetic
            # '20173',    # FI5 : family relationship calculation
            # '20175',    # FI6 : conditional arithmetic
            # '20177',    # FI7 : synonym
            # '20179',    # FI8 : chained arithmetic
            # '20181',    # FI9 : concept interpolation
            # '20183',    # FI10 : arithmetic sequence recognition
            # '20185',    # FI11 : antonym
            # '20187',    # FI12 : square sequence recognition
            # '20189',    # FI13 : subset inclusion logic
            # '20192',    # Number of fluid intelligence questions attempted within time limit
            # '20193',    # FI14 : alphanumeric substitution
            # '20242',    # Fluid intelligence completion status
            # --------- symbol digit matches -----------------------
            '20159',	# Number of symbol digit matches made correctly
            '20195',	# Number of symbol digit matches attempted
            '20196',	# First code array presented
            '20198',	# Test array presented
            '20200',	# Values wanted
            '20229',	# Values entered
            '20230',	# Duration to entering symbol choice
            '20245',	# Symbol digit completion status
            # '20137',	# When symbol digit substitution test completed. xxxxxxxxx
            # ------------- Pairs matching task -------------------
            '20129',    # Number of columns displayed in round
            '20130',    # Number of rows displayed in round
            '20131',    # Number of correct matches in round
            '20132',    # Number of incorrect matches in round
            # '20133',    # Time to complete round
            # '20134',    # When pairs test completed
            '20244',    # Pairs matching completion status

            # ---------- Numeric memory task ---------------------
            # '20138',    # When numeric memory test completed. xxxxxxxxx
            '20240',    # Maximum digits remembered correctly
        ]
        cognitive_online_exhdr = []
        cognitive_online_excat = []

        return cognitive_online_incat, \
               cognitive_online_inhdr, \
               cognitive_online_exhdr, \
               cognitive_online_excat 

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Mental parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_mental_params(
        self,
    ):
        mental_incat = [
            # Anxiety/Depression
            '138',	# Depression
            '140',	# Anxiety
            # '141',	# Addictions
            # '142',	# Alcohol use
            # '143',	# Cannabis use
            # '144',	# Unusual and psychotic experiences
            # '145',	# Traumatic events
            # '146',	# Self-harm behaviours
            '147',	# Happiness and subjective well-being
            # '137',	# Mental distress
            # '139',	# Mania
        ]
        # Based on BMC Paper: Rongtao Jiang, 2022
        mental_inhdr = [
             # ---------- Neuroticism ----------
            '20127',    # Neuroticism
            '1920',     # Mood swings
            '1930',     # Miserableness
            '1940', 	# Irritability
            '1950',	    # Sensitivity / hurt feelings
            '1960', 	# Fed-up feelings
            '1970', 	# Nervous feelings
            '1980', 	# Worrier / anxious feelings
            '1990', 	# Tense / 'highly strung'
            '2000',	    # Worry too long after embarrassment
            '2010', 	# Suffer from 'nerves'
            '2020',	    # Loneliness, isolation
            '2030',	    # Guilty feelings
            # ---------- Subjective well-being ----------
            # '20458',	# General happiness	- Category: 147 Happiness and subjective well-being  
            # '20459',	# General happiness with own health	- Category: 147 Happiness and subjective well-being
            # '20460',	# Belief that own life is meaningful
            
            # ---------- Life satisfaction ----------
            '4526',	    # Happiness
            '4559',     # Family relationship satisfaction
            '4537',	    # Work/job satisfaction
            '4548',	    # Health satisfaction
            '4570',	    # Friendships satisfaction
            '4581',	    # Financial situation satisfaction

            # ---------- Depression ----------
            # '20507',	# Recent feelings of inadequacy
            # '20508',	# trouble concentrating on things
            # '20510',	# Recent feelings of depression
            # '20511',	# Recent poor appetite or overeating
            # '20513',	# Recent thoughts of suicide or self-harm
            # '20514',	# Recent lack of interest or pleasure in doing things
            # '20517',	# Trouble falling or staying asleep, or sleeping too much
            # '20518',	# Recent changes in speed/amount of moving or speaking
            # '20519',	# Recent feelings of tiredness or low energy
            # ---------- Anxiety ----------
            # '20505',	# Recent easy annoyance or irritability
            # '20506',	# Recent feelings or nervousness or anxiety
            # '20509',	# Recent inability to stop or control worrying
            # '20512',	# Recent feelings of foreboding
            # '20515',	# Recent trouble relaxing
            # '20516',	# Recent restlessness
            # '20520',	# Recent worrying too much about different things'
            # ---------- CIDI depression ----------
            # '20441',	# Ever had prolonged loss of interest in normal activities
            # '20446',	# Ever had prolonged feelings of sadness or depression
            # '20449',	# Feelings of tiredness during worst episode of depression
            # '20450',	# Feelings of worthlessness during worst period of depression
            # '20437',	# Thoughts of death during worst depression
            # '20435',	# Difficulty concentrating during worst depression
            # '20532',	# Did your sleep change?
            # '20536',	# Weight change during worst episode of depression
            # ------------------------------------------------------------------
            # '20499',	#Ever sought or received professional help for mental distress
            # '20500',	#Ever suffered mental distress preventing usual activities
            # '20544',	#Mental health problems ever diagnosed by a professional
            # '20547',	#Activities undertaken to treat depression
            # '20433',	#Age at first episode of depression
            # '20434',	#Age at last episode of depression
            # '20445',	#Depression possibly related to childbirth
            # '20447',	#Depression possibly related to stressful or traumatic event
            # '20438',	#Duration of worst depression
            # '20436',	#Fraction of day affected during worst episode of depression
            # '20439',	#Frequency of depressed days during worst episode of depression
            # '20440',	#Impact on normal roles during worst period of depression
            # '20442',	#Lifetime number of depressed periods
            # '20448',	#Professional informed about depression
            # '20534',	#Sleeping too much
            # '20546',	#Substances taken for depression
            # '20533',	#Trouble falling asleep
            # '20535',    #Waking too early
            # '20502',	#Ever had period extreme irritability
            # '20501',	#Ever had period of mania / excitability
            # '20492',	#Longest period of mania or irritability
            # '20548',	#Manifestations of mania or irritability
            # '20493',	#Severity of problems due to mania or irritability
            # 20550	Activities undertaken to treat anxiety
            # 20419	Difficulty concentrating during worst period of anxiety
            # 20541	Difficulty stopping worrying during worst period of anxiety
            # 20429	Easily tired during worst period of anxiety
            # 20421	Ever felt worried, tense, or anxious for most of a month or longer
            # 20425	Ever worried more than most people would in similar situation
            # 20537	Frequency of difficulty controlling worry during worst period of anxiety
            # 20539	Frequency of inability to stop worrying during worst period of anxiety
            # 20427	Frequent trouble falling or staying asleep during worst period of anxiety
            # 20418	Impact on normal roles during worst period of anxiety
            # 20423	Keyed up or on edge during worst period of anxiety
            # 20420	Longest period spent worried or anxious
            # 20422	More irritable than usual during worst period of anxiety
            # 20540	Multiple worries during worst period of anxiety
            # 20543	Number of things worried about during worst period of anxiety
            # 20428	Professional informed about anxiety
            # 20426	Restless during period of worst anxiety
            # 20542	Stronger worrying (than other people) during period of worst anxiety
            # 20549	Substances taken for anxiety
            # 20417	Tense, sore, or aching muscles during worst period of anxiety
            # 20538	Worried most days during period of worst anxiety
            # 20552	Behavioural and miscellaneous addictions
            # 20431	Ever addicted to a behaviour or miscellanous
            # 20406	Ever addicted to alcohol
            # 20401	Ever addicted to any substance or behaviour
            # 20456	Ever addicted to illicit or recreational drugs
            # 20457	Ongoing addiction or dependence on illicit or recreational drugs
            # 20503	Ever addicted to prescription or over-the-counter medication
            # 20504	Ongoing addiction or dependence to over-the-counter medication
            # 20551	Substance of prescription or over-the-counter medication addiction
            # 20404	Ever physically dependent on alcohol
            # 20415	Ongoing addiction to alcohol
            # 20432	Ongoing behavioural or miscellanous addiction
            # 20414	Frequency of drinking alcohol
            # 20403	Amount of alcohol drunk on a typical drinking day
            # 20416	Frequency of consuming six or more units of alcohol
            # 20413	Frequency of inability to cease drinking in last year
            # 20407	Frequency of failure to fulfil normal expectations due to drinking alcohol in last year
            # 20412	Frequency of needing morning drink of alcohol after heavy drinking session in last year
            # 20409	Frequency of feeling guilt or remorse after drinking alcohol in last year
            # 20408	Frequency of memory loss due to drinking alcohol in last year
            # 20411	Ever been injured or injured someone else through drinking alcohol
            # 20405	Ever had known person concerned about, or recommend reduction of, alcohol consumption
            # 20410	Age when known person last commented about drinking habits
            # 20455	Age when last took cannabis
            # 20453	Ever taken cannabis
            # 20454	Maximum frequency of taking cannabis
            # 20461	Age when first had unusual or psychotic experience
            # 20462	Distress caused by unusual or psychotic experiences
            # 20468	Ever believed in an un-real conspiracy against self
            # 20474	Ever believed in un-real communications or signs
            # 20463	Ever heard an un-real voice
            # 20466	Ever prescribed a medication for unusual or psychotic experiences
            # 20471	Ever seen an un-real vision
            # 20477	Ever talked to a health professional about unusual or psychotic experiences
            # 20467	Frequency of unusual or psychotic experiences in past year
            # 20470	Number of times believed in an un-real conspiracy against self
            # 20476	Number of times believed in un-real communications or signs
            # 20465	Number of times heard an un-real voice
            # 20473	Number of times seen an un-real vision
            # 20489	Felt loved as a child
            # 20488	Physically abused by family as a child
            # 20487	Felt hated by family member as a child
            # 20490	Sexually molested as a child
            # 20491	Someone to take to doctor when needed as a child
            # 20522	Been in a confiding relationship as an adult
            # 20523	Physical violence by partner or ex-partner as an adult
            # 20521	Belittlement by partner or ex-partner as an adult
            # 20524	Sexual interference by partner or ex-partner without consent as an adult
            # 20525	Able to pay rent/mortgage as an adult
            # 20531	Victim of sexual assault
            # 20529	Victim of physically violent crime
            # 20526	Been in serious accident believed to be life-threatening
            # 20530	Witnessed sudden violent death
            # 20528	Diagnosed with life-threatening illness
            # 20527	Been involved in combat or exposed to war-zone
            # 20497	Repeated disturbing thoughts of stressful experience in past month
            # 20498	Felt very upset when reminded of stressful experience in past month
            # 20495	Avoided activities or situations because of previous stressful experience in past month
            # 20496	Felt distant from other people in past month
            # 20494	Felt irritable or had angry outbursts in past month
            # 20479	Ever thought that life not worth living
            # 20485	Ever contemplated self-harm
            # 20486	Contemplated self-harm in past year
            # 20480	Ever self-harmed
            # 20482	Number of times self-harmed
            # 20481	Self-harmed in past year
            # 20553	Methods of self-harm used
            # 20554	Actions taken following self-harm
            # 20483	Ever attempted suicide
            # 20484	Attempted suicide in past year
            # 20458	General happiness
            # 20459	General happiness with own health
            # 20460	Belief that own life is meaningful
        ]
        mental_exhdr = []
        mental_excat = []

        return mental_incat, \
               mental_inhdr, \
               mental_exhdr, \
               mental_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Baseline Characteristics parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_baseline_params(
        self,
    ):
        baseline_incat = []
        baseline_inhdr = [
            '189',	    # Townsend deprivation index at recruitment
            '21022',	# Age at recruitment
            '33',	    # Date of birth	#
            '52',	    # Month of birth
            '34',	    # Year of birth
            '31',	    # Sex
        ]
        baseline_exhdr = []
        baseline_excat = []

        return baseline_incat, \
               baseline_inhdr, \
               baseline_exhdr, \
               baseline_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Ongoing characteristics parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_ongoing_params(
        self,
    ):
        ongoing_incat = []
        ongoing_inhdr = [
            # '190',  	# Reason lost to follow-up
            # '191',	# Date lost to follow-up
            # '20005',	# Email access
            # '110007',	# Newsletter communications, date sent
        ]
        ongoing_exhdr = []
        ongoing_excat = []

        return ongoing_incat, \
               ongoing_inhdr, \
               ongoing_exhdr, \
               ongoing_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Physiological Measures parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_biological_sample_params(
        self,
    ):
        biological_sample_incat = [
            # '100002',   # Blood sample collection
            # '100096',   # Saliva sample collection
            # '100095',   # Urine sample collection
        ]
        biological_sample_inhdr = []
        biological_sample_exhdr = []
        biological_sample_excat = []

        return biological_sample_incat, \
               biological_sample_inhdr, \
               biological_sample_exhdr, \
               biological_sample_excat


# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Physiological Measures parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_physical_measures_params(
        self,
    ):
        physical_measures_incat = [
            '100011',   # Blood pressure
            # '101',      # Carotid ultrasound
            # '100007',   # Arterial stiffness
            # '100049',   # Hearing test
            # '100013',   # Eye measures
            # '100018',   # Bone-densitometry of heel
            # '100020',   # Spirometry
            '104',      # ECG at rest, 12-lead
            '100012',   # ECG during exercise
        ]
        physical_measures_inhdr = []
        physical_measures_exhdr = []
        physical_measures_excat = []

        return physical_measures_incat, \
               physical_measures_inhdr, \
               physical_measures_exhdr, \
               physical_measures_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Medical history parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_medical_history_params(
        self,
    ):
        medical_incat = [
            # '100042', # General health
            # '100037', # Breathing
            '100038', # Claudication and peripheral artery disease	
            '100048', # Pain
            '100039', # Chest pain
            '100047', # Operations
            '100044', # Medical conditions
            '100045', # Medication
        ]
        medical_inhdr = []
        medical_exhdr = []
        medical_excat = []

        return medical_incat, \
               medical_inhdr, \
               medical_exhdr, \
               medical_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define hospital parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_hospital_params(self):
        hospital_incat = [
            # '2002', # Contains summary fields relating to diagnoses --
                    # made during hospital inpatient admissions.
                    # Including main and secondary diagnoses(ICD10 & ICD9 codes)
            # '2006', # Record-level access --
                    # -- Consists of 7 interrelated database tables: 
                    # HESIN, HESIN_DIAG, HESIN_OPER, HESIN_CRITICAL, HESIN_PSYCH, 
                    # HESIN_MATERNITY and HESIN_DELIVERY.
            # '2005', # Contains summary fields relating to operations/procedures
                    # performed during hospital inpatient admissions.
                    # The Office of Population Censuses and Surveys Classification
                    # of Interventions and Procedures (OPCS-3 & OPCS-4). 
        ]
        hospital_inhdr = [
            # ----- Summary Diagnoses ------ #
            '41270',	# Diagnoses - ICD10
            '41280',	# Date of first in-patient diagnosis - ICD10
            '41202',	# Diagnoses - main ICD10
            '41262',	# Date of first in-patient diagnosis - main ICD10
            '41204',	# Diagnoses - secondary ICD10
            '41201',	# External causes - ICD10
        ]
        hospital_exhdr = []
        hospital_excat = []

        return hospital_incat, \
               hospital_inhdr, \
               hospital_exhdr, \
               hospital_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Mental health conditions ever diagnosed by a professional Field on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_mental_health_by_professional(self):
        mental_health_incat = []
        mental_health_inhdr = [
            '29000', # Mental health conditions ever diagnosed by a professional
        ]
        mental_health_exhdr = []
        mental_health_excat = []

        return mental_health_incat, \
               mental_health_inhdr, \
               mental_health_exhdr, \
               mental_health_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define self-report parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_selfreport_params(self):

        selfreport_incat = []
        selfreport_inhdr = [
            '20002', # Non-cancer illness code, self-reported
            # '20009', # Interpolated Age of participant when non-cancer illness 
            #          # first diagnosed
            # '20008', # Interpolated Year when non-cancer illness first diagnosed
            # '87',	 # Non-cancer illness year/age first occurred
            # '2956',	 # General pain for 3+ months
            # '135',	 # Number of self-reported non-cancer illnesses
            # '3404',	 # Neck/shoulder pain for 3+ months
            # '3799',  # Headaches for 3+ months
            # '20013', # Method of recording time when non-cancer illness first diagnosed
        ]
        selfreport_exhdr = []
        selfreport_excat = []

        return selfreport_incat, \
               selfreport_inhdr, \
               selfreport_exhdr, \
               selfreport_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define death register parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_death_params(self):
        deathregister_incat = [
             '100093', # Death register
                      # Contains coded data on the cause of death and Dates
        ]
        deathregister_inhdr = []
        deathregister_exhdr = []
        deathregister_excat = []

        return deathregister_incat, \
               deathregister_inhdr, \
               deathregister_exhdr, \
               deathregister_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define death register parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_first_occurrences_params(self):
        
        if self.population_name == "stroke":
            first_occurrences_incat = []
            first_occurrences_inhdr = [
            # Category 2409 - Circulatory system disorders-First occurrences
                '131366',   # Date I63 first reported (cerebral infarction)
                '131362',   # Date I61 first reported (intracerebral haemorrhage)
            ]
            first_occurrences_exhdr = []
            first_occurrences_excat = []
        
        elif self.population_name == "parkinson":
            first_occurrences_incat = []
            first_occurrences_inhdr = [
                # Category 2406
                '131022', # Date G20 first reported (parkinson's disease)
                        # Nervous system disorders
                '131023', # Source of report of G20 (parkinson's disease)
                        # Nervous system disorders
                '131024', # Date G21 first reported (secondary parkinsonism)
                        # Nervous system disorders  
                '131026', # Date G22 first reported (parkinsonism in diseases classified elsewhere)
                        # Nervous system disorders
                '131025', # Source of report of G21 (secondary parkinsonism)
                        # Nervous system disorders  
                '131027', # Source of report of G22 (parkinsonism in diseases classified elsewhere)
                        # Nervous system disorders
                '26261',  # Enhanced PRS for parkinson's disease (PD)
                        # Enhanced PRS  
                '26260',  # Standard PRS for parkinson's disease (PD)
                        # Standard PRS
            ]
            first_occurrences_exhdr = []
            first_occurrences_excat = []
        
        elif self.population_name == "depression":
            first_occurrences_incat = []
            first_occurrences_inhdr = [
                # Category 2405
                '130894',	# Date F32 first reported (depressive episode)
                '130895',	# Source of report of F32 (depressive episode)
                '130896',	# Date F33 first reported (recurrent depressive disorder)
                '130897',	# Source of report of F33 (recurrent depressive disorder)
            ]
            first_occurrences_exhdr = []
            first_occurrences_excat = []

        return first_occurrences_incat, \
               first_occurrences_inhdr, \
               first_occurrences_exhdr, \
               first_occurrences_excat
# -----------------------------------------------------------------------------#
# --------------------------------#
# Define Walking parameters on UK Biobank 
# --------------------------------#
# -----------------------------------------------------------------------------#
    # def get_walking_params(
    #     self,
    # ):
    #     walking_incat = []
    #     walking_inhdr = [ 
    #         # '864',      # Number of days/week walked 10+ minutes
    #         # '874',	  # Duration of walks
    #         # '6162',	  # Types of transport used (excluding work)
    #         # '924',	  # Usual walking pace
    #         # '971',	  # Frequency of walking for pleasure in last 4 weeks
    #         # '943',      # Frequency of stair climbing in last 4 weeks
    #         # '981',	  # Duration walking for pleasure
    #         # '1097'      # 1Duration of vigorous physical activity (pilot)
    #     ]
    #     walking_exhdr = []
    #     walking_excat = []

    #     return walking_incat, \
    #            walking_inhdr, \
    #            walking_exhdr, \
    #            walking_excat

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define ukbb_parser extra flags 
# for:
# - MRI flag: --img_subs_only or not
# - long_names flag: to replace datafield numbers in column names with datafield titles
# - fillna flag: To fill blank cells with 'NaN'
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_ukbb_parser_flags_params(
        self,
    ):
        """ 
        Get the list of flags to be included to ukbb_parser command
            -----------
            Parameters
            -----------
            mri : binary
                The binary values of MRI data status for subjects with/without MRI
                0 --> non_MRI data (subjects without MRI data)
                1 --> MRI data (subjects with MRI data)
            -----------
            Returns
            -----------
            ukbb_parser_flags : list of str
                The list of flags to be included to ukbb_parser command
        """
        if self.mri == 0:
            ukbb_parser_flags = [
            # ' --long_names',
            ' --fillna NaN',
            ]
        elif self.mri == 1:
            ukbb_parser_flags = [
                # ' --long_names',
                ' --fillna NaN',
                ' --img_subs_only'
            ]

        return ukbb_parser_flags

# -----------------------------------------------------------------------------#
# --------------------------------#
# Define ukbb_parser command
# Based on different functions/ parameteres
# To run on other .py file (ukbb_parser_run.py)
# --------------------------------#
# -----------------------------------------------------------------------------#
    def get_ukbb_parser_cmd(
        self,
    ):
        """ 
        Get the ukbb string command of Healthy population to be included to
        the ukbb_parser command.            
            -----------
            Parameters
            -----------
            No inputs
                all parameters and function are using from class UkbbParams
            -----------
            Returns
            -----------
            data_field_list : list of str
                The ukb string command to be included to ukbb_parser command
        """
        data_field_list = ""

        # -------- Healthy, stroke or Parkinson's disease Populations --------#
        # Get the list of Population Field IDs
        if self.ishealthy == 1 :
            icd10_excon, \
            icd10_incon =  self.get_healthy_params()
        else:
            if self.population_name in "stroke":
                icd10_excon, \
                icd10_incon =  self.get_stroke_params()

                # ------------- Stroke Outcome --> Category 43 ------------- #
                # Get the list of Field, Category IDs to be included/excluded 
                # to/from ukbb_parser command
                stroke_outcome_incat, \
                stroke_outcome_inhdr, \
                stroke_outcome_exhdr, \
                stroke_outcome_excat = self.get_stroke_outcomes()
                # ------ Stroke ------- #
                # Get the list of Field, Category IDs to be included/excluded 
                # to/from ukbb_parser command
                first_occurrences_incat, \
                first_occurrences_inhdr, \
                first_occurrences_exhdr, \
                first_occurrences_excat = self.get_first_occurrences_params()
                
            elif self.population_name == "parkinson":
                icd10_excon, \
                icd10_incon =  self.get_parkinson_params()

                # ------ Parkinson's disease Outcome --> Category 50 ------- #
                # Get the list of Field, Category IDs to be included/excluded 
                # to/from ukbb_parser command
                pd_outcome_incat, \
                pd_outcome_inhdr, \
                pd_outcome_exhdr, \
                pd_outcome_excat = self.get_parkinson_outcomes()
                # ------ Parkinson's disease Nervouse System Disorders ------- #
                # Get the list of Field, Category IDs to be included/excluded 
                # to/from ukbb_parser command
                first_occurrences_incat, \
                first_occurrences_inhdr, \
                first_occurrences_exhdr, \
                first_occurrences_excat = self.get_first_occurrences_params()
            elif self.population_name == "depression":
                icd10_excon, \
                icd10_incon =  self.get_depression_params()

                # ------ Depression Outcome ------- #
                # Get the list of Field, Category IDs to be included/excluded 
                # to/from ukbb_parser command
                depression_outcome_incat, \
                depression_outcome_inhdr, \
                depression_outcome_exhdr, \
                depression_outcome_excat = self.get_depression_outcomes()
                # ------ Depression First Occurrences ------- #
                # Get the list of Field, Category IDs to be included/excluded 
                # to/from ukbb_parser command
                first_occurrences_incat, \
                first_occurrences_inhdr, \
                first_occurrences_exhdr, \
                first_occurrences_excat = self.get_first_occurrences_params()
                
                # ------ Mental health conditions ever diagnosed by a professional Field on UK Biobank 
                mental_health_incat, \
                mental_health_inhdr, \
                mental_health_exhdr, \
                mental_health_excat = self.get_mental_health_by_professional()

        # --------- Handgrip Strength (HGS) as the Motor Performance ---------#
        # Get the list of Motor Performance Field IDs
        if self.motor_type in ["hgs", "handgrip_strength"]:
            motor_incat, \
            motor_inhdr, \
            motor_exhdr, \
            motor_excat = self.get_motor_params()

        # ------ AssessmentCentre --> Category 100024 (Reception) -----------#
        # Get the list of Assessment Field IDs
        assessment_incat, \
        assessment_inhdr, \
        assessment_exhdr, \
        assessment_excat = self.get_assessment_params()
 
        # ---------- List of ukbb_parser flags with (MRI/Non-MRI) ----------- #
        # Get the list of flags to be included to ukbb_parser command
        ukbb_parser_flags = self.get_ukbb_parser_flags_params()
        
        # ---- Demographics --> Category 1001 (Baseline characteristics ) ----#
        # Get the list of Field IDs
        demographic_incat, \
        demographic_inhdr, \
        demographic_exhdr, \
        demographic_excat = self.get_demographic_params()

        # ------------- SocioDemographics --> Category 100062 --------------- #
        # Get the list of Field IDs        
        socio_incat, \
        socio_inhdr, \
        socio_exhdr, \
        socio_excat = self.get_sociodemographics_param()
        
        # ------------- Lifestyle --> Category 100050 --------------- #
        # Get the list of Field IDs        
        lifestyle_incat, \
        lifestyle_inhdr, \
        lifestyle_exhdr, \
        lifestyle_excat = self.get_lifestyle_params()
        
        # ------------- Body size Meseares --> Category 100010 --------------- #
        # Get the list of Field IDs        
        body_size_incat, \
        body_size_inhdr, \
        body_size_exhdr, \
        body_size_excat, \
        body_size_impedance_incat, \
        body_size_impedance_inhdr, \
        body_size_impedance_exhdr, \
        body_size_impedance_excat = self.get_body_measures_params()

        # ------------- cognitive Function --> Category 100026 -------------- #
        # Get the list of Field IDs       
        cognitive_incat, \
        cognitive_inhdr, \
        cognitive_exhdr, \
        cognitive_excat = self.get_cognitive_clinic_params()

        # ------- Cognitive function online --> Category 116 -------------- #
        # Get the list of Field IDs      
        cognitive_online_incat, \
        cognitive_online_inhdr, \
        cognitive_online_exhdr, \
        cognitive_online_excat = self.get_cognitive_online_params()

        # ------- Mental online --> Category 136 -------------- #
        # Get the list of Field IDs     
        mental_incat, \
        mental_inhdr, \
        mental_exhdr, \
        mental_excat = self.get_mental_params()

        # ------- Baseline Characteristics --> Category 100094 -------------- #
        # Get the list of Field IDs     
        baseline_incat, \
        baseline_inhdr, \
        baseline_exhdr, \
        baseline_excat = self.get_baseline_params()
        
        # ------ Hospital Inpatient ------ Category 2000
        # Get the list of Field, Category IDs to be included/excluded 
        # to/from ukbb_parser command
        hospital_incat, \
        hospital_inhdr, \
        hospital_exhdr, \
        hospital_excat = self.get_hospital_params()

        # ------ Self-Report ------
        # Get the list of Field IDs to be included/excluded 
        # to/from ukbb_parser command
        selfreport_incat, \
        selfreport_inhdr, \
        selfreport_exhdr, \
        selfreport_excat = self.get_selfreport_params()

        # ------ Death register ------ Category 100093
        # Get the list of Field, Category IDs to be included/excluded 
        # to/from ukbb_parser command
        deathregister_incat, \
        deathregister_inhdr, \
        deathregister_exhdr, \
        deathregister_excat = self.get_death_params()

        # -------------------------------------------------------------------- #
        # --------------------------------#
        # Define ukbb_parser command
        # Based on different functions/ parameteres
        # To run on other .py file (ukbb_parser_run.py)
        # The variable name is >> data_field_list <<
        # --------------------------------#
        data_field_list = ""

        # ------ Add ICD-10 codes to ukbb string --------------------------- #
        for n_icd10 in range(len(icd10_incon)):
            data_field_list = f'{data_field_list}' \
            ' --incon ' f'{icd10_incon[n_icd10]}'

        for n_icd10 in range(len(icd10_excon)):
            data_field_list = f'{data_field_list}' \
            ' --excon ' f'{icd10_excon[n_icd10]}'

        # ------ Add (--inhdr) Motor Performance Field IDs ------------------- #
        for n_motor in range(len(motor_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{motor_incat[n_motor]}'
        
        for n_motor in range(len(motor_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{motor_inhdr[n_motor]}'

        for n_motor in range(len(motor_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{motor_exhdr[n_motor]}'

        for n_motor in range(len(motor_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{motor_excat[n_motor]}'

        # -------- stroke or Parkinson's disease Populations outcomes --------#
        if self.population_name in "stroke":
            # ------ Add Stroke Outcome ------ Category 43 --------------- #
            # Get the list of Field, Category IDs to be included/excluded 
            # to/from ukbb_parser command        
            for n_inhdr in range(len(stroke_outcome_inhdr)):
                data_field_list = f'{data_field_list}' \
                ' --inhdr ' f'{stroke_outcome_inhdr[n_inhdr]}'

            for n_exhdr in range(len(stroke_outcome_exhdr)):
                data_field_list = f'{data_field_list}' \
                ' --exhdr ' f'{stroke_outcome_exhdr[n_exhdr]}'

            for n_incat in range(len(stroke_outcome_incat)):
                data_field_list = f'{data_field_list}' \
                ' --incat ' f'{stroke_outcome_incat[n_incat]}'
            
            for n_excat in range(len(stroke_outcome_excat)):
                data_field_list = f'{data_field_list}' \
                ' --excat ' f'{stroke_outcome_excat[n_excat]}'
        elif self.population_name == "parkinson":
            # ------ Add Parkinson Outcome ------ Category 50 ------------ #
            # Get the list of Field, Category IDs to be included/excluded 
            # to ukbb_parser command        
            for n_inhdr in range(len(pd_outcome_inhdr)):
                data_field_list = f'{data_field_list}' \
                ' --inhdr ' f'{pd_outcome_inhdr[n_inhdr]}'

            for n_exhdr in range(len(pd_outcome_exhdr)):
                data_field_list = f'{data_field_list}' \
                ' --exhdr ' f'{pd_outcome_exhdr[n_exhdr]}'

            for n_incat in range(len(pd_outcome_incat)):
                data_field_list = f'{data_field_list}' \
                ' --incat ' f'{pd_outcome_incat[n_incat]}'
            
            for n_excat in range(len(pd_outcome_excat)):
                data_field_list = f'{data_field_list}' \
                ' --excat ' f'{pd_outcome_excat[n_excat]}'
        elif self.population_name == "depression":
            # ------ Add Depression Outcome ------ #
            # Get the list of Field, Category IDs to be included/excluded 
            # to ukbb_parser command        
            for n_inhdr in range(len(depression_outcome_inhdr)):
                data_field_list = f'{data_field_list}' \
                ' --inhdr ' f'{depression_outcome_inhdr[n_inhdr]}'

            for n_exhdr in range(len(depression_outcome_exhdr)):
                data_field_list = f'{data_field_list}' \
                ' --exhdr ' f'{depression_outcome_exhdr[n_exhdr]}'

            for n_incat in range(len(depression_outcome_incat)):
                data_field_list = f'{data_field_list}' \
                ' --incat ' f'{depression_outcome_incat[n_incat]}'
            
            for n_excat in range(len(depression_outcome_excat)):
                data_field_list = f'{data_field_list}' \
                ' --excat ' f'{depression_outcome_excat[n_excat]}'
        # ------ Add (--inhdr) Assessment Feild IDs -------------------------- #
        for n_inhdr in range(len(assessment_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{assessment_incat[n_inhdr]}'

        for n_inhdr in range(len(assessment_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{assessment_inhdr[n_inhdr]}'

        for n_inhdr in range(len(assessment_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{assessment_exhdr[n_inhdr]}'

        for n_inhdr in range(len(assessment_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{assessment_excat[n_inhdr]}'
    
        # ---- Add Demographics --> Category 1001 (Baseline characteristics )-#
        for n_inhdr in range(len(demographic_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{demographic_inhdr[n_inhdr]}'

        for n_exhdr in range(len(demographic_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{demographic_exhdr[n_exhdr]}'

        for n_incat in range(len(demographic_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{demographic_incat[n_incat]}'

        for n_excat in range(len(demographic_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{demographic_excat[n_excat]}'

        # ------------- Add Lifestyle --> Category 100050 --------------- #
        for n_inhdr in range(len(lifestyle_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{lifestyle_inhdr[n_inhdr]}'

        for n_exhdr in range(len(lifestyle_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{lifestyle_exhdr[n_exhdr]}'

        for n_incat in range(len(lifestyle_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{lifestyle_incat[n_incat]}'

        for n_excat in range(len(lifestyle_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{lifestyle_excat[n_excat]}'

        # ------------- SocioDemographics --> Category 100062 --------------- #
        for n_inhdr in range(len(socio_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{socio_inhdr[n_inhdr]}'

        for n_exhdr in range(len(socio_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{socio_exhdr[n_exhdr]}'

        for n_incat in range(len(socio_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{socio_incat[n_incat]}'

        for n_excat in range(len(socio_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{socio_excat[n_excat]}'

        # ------------- Body size Meseares --> Category 100010 --------------- #
        for n_inhdr in range(len(body_size_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{body_size_inhdr[n_inhdr]}'

        for n_exhdr in range(len(body_size_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{body_size_exhdr[n_exhdr]}'

        for n_incat in range(len(body_size_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{body_size_incat[n_incat]}'

        for n_excat in range(len(body_size_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{body_size_excat[n_excat]}'

        # ------------- Body size impedance Meseares --> Category 100010 ----- #
        for n_inhdr in range(len(body_size_impedance_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{body_size_impedance_inhdr[n_inhdr]}'

        for n_exhdr in range(len(body_size_impedance_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{body_size_impedance_exhdr[n_exhdr]}'

        for n_incat in range(len(body_size_impedance_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{body_size_impedance_incat[n_incat]}'

        for n_excat in range(len(body_size_impedance_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{body_size_impedance_excat[n_excat]}'

        # ------------- cognitive Function --> Category 100026 -------------- #
        for n_inhdr in range(len(cognitive_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{cognitive_inhdr[n_inhdr]}'

        for n_exhdr in range(len(cognitive_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{cognitive_exhdr[n_exhdr]}'

        for n_incat in range(len(cognitive_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{cognitive_incat[n_incat]}'

        for n_excat in range(len(cognitive_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{cognitive_excat[n_excat]}'

        # ------- Cognitive function online --> Category 116 -------------- #
        for n_inhdr in range(len(cognitive_online_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{cognitive_online_inhdr[n_inhdr]}'

        for n_exhdr in range(len(cognitive_online_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{cognitive_online_exhdr[n_exhdr]}'

        for n_incat in range(len(cognitive_online_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{cognitive_online_incat[n_incat]}'

        for n_excat in range(len(cognitive_online_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{cognitive_online_excat[n_excat]}'

        # ------- Mental online --> Category 136 -------------- #
        for n_inhdr in range(len(mental_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{mental_inhdr[n_inhdr]}'

        for n_exhdr in range(len(mental_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{mental_exhdr[n_exhdr]}'

        for n_incat in range(len(mental_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{mental_incat[n_incat]}'

        for n_excat in range(len(mental_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{mental_excat[n_excat]}'

        # ------- Baseline Characteristics --> Category 100094 -------------- #
        for n_inhdr in range(len(baseline_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{baseline_inhdr[n_inhdr]}'

        for n_exhdr in range(len(baseline_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{baseline_exhdr[n_exhdr]}'

        for n_incat in range(len(baseline_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{baseline_incat[n_incat]}'

        for n_excat in range(len(baseline_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{baseline_excat[n_excat]}'

        #  # ------- Hospital inpatient--> Category 2000 -------------- #
        for n_inhdr in range(len(hospital_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{hospital_inhdr[n_inhdr]}'

        for n_exhdr in range(len(hospital_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{hospital_exhdr[n_exhdr]}'

        for n_incat in range(len(hospital_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{hospital_incat[n_incat]}'

        for n_excat in range(len(hospital_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{hospital_excat[n_excat]}'

        # -------- Mental Health by Professional -------- #
        if self.population_name == "depression":
            for n_inhdr in range(len(mental_health_inhdr)):
                data_field_list = f'{data_field_list}' \
                ' --inhdr ' f'{mental_health_inhdr[n_inhdr]}'

            for n_exhdr in range(len(mental_health_exhdr)):
                data_field_list = f'{data_field_list}' \
                ' --exhdr ' f'{mental_health_exhdr[n_exhdr]}'

            for n_incat in range(len(mental_health_incat)):
                data_field_list = f'{data_field_list}' \
                ' --incat ' f'{mental_health_incat[n_incat]}'

            for n_excat in range(len(mental_health_excat)):
                data_field_list = f'{data_field_list}' \
                ' --excat ' f'{mental_health_excat[n_excat]}'
            
        # -------- Death Register ------- #
        for n_inhdr in range(len(deathregister_inhdr)):
            data_field_list = f'{data_field_list}' \
            ' --inhdr ' f'{deathregister_inhdr[n_inhdr]}'

        for n_exhdr in range(len(deathregister_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{deathregister_exhdr[n_exhdr]}'

        for n_incat in range(len(deathregister_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{deathregister_incat[n_incat]}'

        for n_excat in range(len(deathregister_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{deathregister_excat[n_excat]}'
        #-------- First Occurrences ------- #
        if self.population_name in ["stroke", "parkinson", "depression"]:
            for n_inhdr in range(len(first_occurrences_inhdr)):
                data_field_list = f'{data_field_list}' \
                ' --inhdr ' f'{first_occurrences_inhdr[n_inhdr]}'

            for n_exhdr in range(len(first_occurrences_exhdr)):
                data_field_list = f'{data_field_list}' \
                ' --exhdr ' f'{first_occurrences_exhdr[n_exhdr]}'

            for n_incat in range(len(first_occurrences_incat)):
                data_field_list = f'{data_field_list}' \
                ' --incat ' f'{first_occurrences_incat[n_incat]}'

            for n_excat in range(len(first_occurrences_excat)):
                data_field_list = f'{data_field_list}' \
                ' --excat ' f'{first_occurrences_excat[n_excat]}'
        #-------- Self-Report medical Condition ------- #
        for n_incat in range(len(selfreport_incat)):
            data_field_list = f'{data_field_list}' \
            ' --incat ' f'{selfreport_incat[n_incat]}'

        for n_insr in range(len(selfreport_inhdr)):
            data_field_list = f'{data_field_list}' \
           ' --inhdr ' f'{selfreport_inhdr[n_insr]}'

        for n_exsr in range(len(selfreport_exhdr)):
            data_field_list = f'{data_field_list}' \
            ' --exhdr ' f'{selfreport_exhdr[n_exsr]}'

        for n_excat in range(len(selfreport_excat)):
            data_field_list = f'{data_field_list}' \
            ' --excat ' f'{selfreport_excat[n_excat]}'

        # -------- Add Flags for ukbb_parser ------- #

        for n_flag in range(len(ukbb_parser_flags)):
                data_field_list = f'{data_field_list}' + \
                                  ukbb_parser_flags[n_flag]

        return data_field_list

