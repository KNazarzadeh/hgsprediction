import math
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from statsmodels.graphics.gofplots import qqplot
from itertools import product
from hgsprediction.load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results
from hgsprediction.save_results.save_prepared_data_for_anova import save_prepare_data_for_anova
from scipy import stats
from scipy.stats import zscore
import statsmodels.formula.api as smf
import researchpy as rp

import matplotlib.pyplot as plt
import seaborn as sns
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
confound_status = sys.argv[5]
n_repeats = sys.argv[6]
n_folds = sys.argv[7]
disorder_cohort = sys.argv[8]
visit_session = sys.argv[9]
n_samples = sys.argv[10]
target = sys.argv[11]
##############################################################################
main_extracted_columns = ["gender", "handedness", "hgs_dominant", "hgs_dominant_side", "hgs_nondominant", "hgs_nondominant_side", "age", "bmi", "height", "waist_to_hip_ratio", "treatment", "disorder_episode", "hgs_target", "hgs", "hgs_predicted", "hgs_delta", "hgs_corrected_predicted", "hgs_corrected_delta"]

df_disorder = pd.DataFrame()
df_control = pd.DataFrame()

disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
    
df_disorder_matched_female, df_mathced_controls_female = load_disorder_matched_samples_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "female",
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
)

df_disorder_matched_male, df_mathced_controls_male = load_disorder_matched_samples_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "male",
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
)


# Assume df_female and df_male are your initial DataFrames
np.random.seed(42)  # For reproducibility

# Function to randomly select 10 subjects for each patient_id
def select_random_samples(df, n=10):
    return df.groupby('1st_pre-depression_patient_id').apply(lambda x: x.sample(n))

# Apply the function to both DataFrames
df_female_sampled = select_random_samples(df_mathced_controls_female)
df_male_sampled = select_random_samples(df_mathced_controls_male)
print("===== Done! End =====")
embed(globals(), locals())

