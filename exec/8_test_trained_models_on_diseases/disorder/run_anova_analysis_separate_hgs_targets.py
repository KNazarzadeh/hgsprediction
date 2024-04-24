import math
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc

from hgsprediction.load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results
from hgsprediction.load_results.load_prepared_data_for_anova import load_prepare_data_for_anova
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
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
# Load data for ANOVA
df = load_prepare_data_for_anova(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
)
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)

df_female = df[df["gender"]=="female"]
df_male = df[df["gender"]=="male"]

print("===== Done! End =====")
embed(globals(), locals())

# for anova_target in ["hgs", "hgs_predicted", "hgs_corrected_predicted", "hgs_delta", "hgs_corrected_delta"]:
for anova_target in ["hgs_corrected_delta"]:
    print(anova_target)
    formula = (
        f'{anova_target} ~ '
        'C(gender) + C(treatment) + C(disorder_episode) +'
        'C(gender):C(treatment) + C(gender):C(disorder_episode) +'
        'C(treatment):C(disorder_episode) '
    )

    model = ols(formula, data=df).fit()
    df_anova_result = sm.stats.anova_lm(model)

    print(df_anova_result)

    # Perform post-hoc tests on significant interactions (Tukey's HSD)
    interaction_groups =  df.disorder_episode.astype(str)    
    comp = mc.MultiComparison(df[f"{anova_target}"], interaction_groups)
    df_post_hoc_result_without_gender = comp.tukeyhsd()
    print(df_post_hoc_result_without_gender.summary())

    interaction_groups =  df.gender.astype(str) + " | " + df.disorder_episode.astype(str)
    comp = mc.MultiComparison(df[f"{anova_target}"], interaction_groups)
    df_post_hoc_result_with_gender = comp.tukeyhsd()
    print(df_post_hoc_result_with_gender.summary())

    print(anova_target)
    print(target)
    print(formula)
    print(population)

    save_disorder_anova_results(
            df,
            df_anova_result,
            df_post_hoc_result_without_gender,
            df_post_hoc_result_with_gender,
            population,
            mri_status,
            session_column,
            model_name,
            feature_type,
            target,
            confound_status,
            n_repeats,
            n_folds,
            n_samples,
            anova_target,
        )

print("===== Done! End =====")
embed(globals(), locals())

