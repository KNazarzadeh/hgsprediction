import math
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats

from hgsprediction.load_results.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results
from hgsprediction.define_features import define_features
from hgsprediction.load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results

import seaborn as sns
import matplotlib.pyplot as plt

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
disorder_cohort = sys.argv[10]
visit_session = sys.argv[11]
n_samples = sys.argv[12]

##############################################################################
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

print("===== Done! =====")
embed(globals(), locals())
##############################################################################

df_anova=pd.concat([df,df_stroke_together])
a = df_anova[["disease", "gender", "delta", "hgs_target", "stroke_cohort"]]
b = a[a["hgs_target"]=="HGS L+R"]
b = b.rename(columns={"disease":"group", "stroke_cohort":"disease_time"})
b["group"].replace(0, "healthy", inplace=True)
b["group"].replace(1, "stroke", inplace=True)
b["gender"].replace(0, "female", inplace=True)
b["gender"].replace(1, "male", inplace=True)
formula = 'delta ~ C(group) + C(disease_time) + C(hgs_target) + C(gender) + C(group):C(disease_time) + C(group):C(hgs_target) + C(group):C(gender) + C(disease_time):C(hgs_target) + C(disease_time):C(gender) + C(hgs_target):C(gender) + C(group):C(disease_time):C(hgs_target) + C(group):C(disease_time):C(gender) + C(group):C(hgs_target):C(gender) + C(disease_time):C(hgs_target):C(gender) + C(group):C(disease_time):C(hgs_target):C(gender)'
model = ols(formula, b).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

print(anova_results)

