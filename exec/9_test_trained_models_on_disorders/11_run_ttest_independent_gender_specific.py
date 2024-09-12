import sys
import os
import numpy as np
import pandas as pd
import math
from scipy.stats import ttest_ind
from hgsprediction.load_results.anova.load_prepared_data_for_anova import load_prepare_data_for_anova

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
firts_event = sys.argv[12]
anova_target = sys.argv[13]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

##############################################################################
# Load data for ANOVA
data = load_prepare_data_for_anova(
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
    firts_event,
)

df = data[["gender", "group", "time_point", anova_target]]

data_female = df[df['gender'] == 0]
data_male = df[df['gender'] == 1]

df_female_pre_control = data_female[data_female['time_point']=="pre-control"]
df_female_post_control = data_female[data_female['time_point']=="post-control"]
df_female_pre_control = df_female_pre_control.reindex(df_female_post_control.index)

df_female_pre_disorder = data_female[data_female['time_point']==f"pre-{population}"]
df_female_post_disorder = data_female[data_female['time_point']==f"post-{population}"]
df_female_pre_disorder = df_female_pre_disorder.reindex(df_female_post_disorder.index)

df_male_pre_control = data_male[data_male['time_point']=="pre-control"]
df_male_post_control = data_male[data_male['time_point']=="post-control"]
df_male_pre_control = df_male_pre_control.reindex(df_male_post_control.index)

df_male_pre_disorder = data_male[data_male['time_point']==f"pre-{population}"]
df_male_post_disorder = data_male[data_male['time_point']==f"post-{population}"]
df_male_pre_disorder = df_male_pre_disorder.reindex(df_male_post_disorder.index)

# Assuming df_post_control and df_pre_control are defined elsewhere
df_female_interaction_control = pd.DataFrame(index=df_female_pre_control.index)
df_female_interaction_disorder = pd.DataFrame(index=df_female_pre_disorder.index)

df_male_interaction_control = pd.DataFrame(index=df_male_pre_control.index)
df_male_interaction_disorder = pd.DataFrame(index=df_male_pre_disorder.index)
# Assuming df_post_control and df_pre_control have the same indices
df_female_interaction_control[f"interaction_{anova_target}"] = df_female_post_control[f"{anova_target}"].values - df_female_pre_control[f"{anova_target}"].values
df_female_interaction_control["group"] = "control"
df_female_interaction_control["time_point"] = "Interaction"
df_female_interaction_control["gender"] = "female"

# Assuming df_post_disorder and df_pre_disorder are defined elsewhere
df_female_interaction_disorder[f"interaction_{anova_target}"] = df_female_post_disorder[f"{anova_target}"].values - df_female_pre_disorder[f"{anova_target}"].values
df_female_interaction_disorder["group"] = f"{population}"
df_female_interaction_disorder["time_point"] = "Interaction"
df_female_interaction_disorder["gender"] = "female"
# Assuming df_post_control and df_pre_control have the same indices
df_male_interaction_control[f"interaction_{anova_target}"] = df_male_post_control[f"{anova_target}"].values - df_male_pre_control[f"{anova_target}"].values
df_male_interaction_control["group"] = "control"
df_male_interaction_control["time_point"] = "Interaction"
df_male_interaction_control["gender"] = "male"

# Assuming df_post_disorder and df_pre_disorder are defined elsewhere
df_male_interaction_disorder[f"interaction_{anova_target}"] = df_male_post_disorder[f"{anova_target}"].values - df_male_pre_disorder[f"{anova_target}"].values
df_male_interaction_disorder["group"] = f"{population}"
df_male_interaction_disorder["time_point"] = "Interaction"
df_male_interaction_disorder["gender"] = "male"

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df_ttest = pd.DataFrame(index=["pre-time_point", "post-time_point", "interaction"], columns=["female", "male"])

stat_pre_female, p_value_pre_female = ttest_ind(df_female_pre_control[f"{anova_target}"], df_female_pre_disorder[f"{anova_target}"])
stat_post_female, p_value_post_female = ttest_ind(df_female_post_control[f"{anova_target}"], df_female_post_disorder[f"{anova_target}"])

stat_pre_male, p_value_pre_male = ttest_ind(df_male_pre_control[f"{anova_target}"], df_male_pre_disorder[f"{anova_target}"])
stat_post_male, p_value_post_male = ttest_ind(df_male_post_control[f"{anova_target}"], df_male_post_disorder[f"{anova_target}"])

df_ttest.loc["pre-time_point", "female"] = p_value_pre_female
df_ttest.loc["post-time_point", "female"] = p_value_post_female
df_ttest.loc["pre-time_point", "male"] = p_value_pre_male
df_ttest.loc["post-time_point", "male"] = p_value_post_male

stat_interaction_female, p_value_interaction_female = ttest_ind(df_female_interaction_control[f"interaction_{anova_target}"], df_female_interaction_disorder[f"interaction_{anova_target}"])
stat_interaction_male, p_value_interaction_male = ttest_ind(df_male_interaction_control[f"interaction_{anova_target}"], df_male_interaction_disorder[f"interaction_{anova_target}"])

df_ttest.loc["interaction", "female"] = p_value_interaction_female
df_ttest.loc["interaction", "male"] = p_value_interaction_male

###############################################################################
print(df_ttest)

print("Female\n MEAN")
print("control= {:.2f}".format(df_female_interaction_control[f'interaction_{anova_target}'].mean()))
print("disease= {:.2f}".format(df_female_interaction_disorder[f'interaction_{anova_target}'].mean()))
print("Female\n SD")
print("control= {:.2f}".format(df_female_interaction_control[f'interaction_{anova_target}'].std()))
print("disease= {:.2f}".format(df_female_interaction_disorder[f'interaction_{anova_target}'].std()))


print("Male\n MEAN")
print("control= {:.2f}".format(df_male_interaction_control[f'interaction_{anova_target}'].mean()))
print("disease= {:.2f}".format(df_male_interaction_disorder[f'interaction_{anova_target}'].mean()))
print("Male\n SD")
print("control= {:.2f}".format(df_male_interaction_control[f'interaction_{anova_target}'].std()))
print("disease= {:.2f}".format(df_male_interaction_disorder[f'interaction_{anova_target}'].std()))

print("===== Done! End =====")
embed(globals(), locals())