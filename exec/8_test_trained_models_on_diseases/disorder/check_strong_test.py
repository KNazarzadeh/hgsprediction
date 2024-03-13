import sys
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
from scipy.stats import ttest_ind


from hgsprediction.load_results.load_disorder_anova_results import load_disorder_anova_results

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
session = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
disorder_cohort = sys.argv[9]
visit_session = sys.argv[10]
n_samples = sys.argv[11]
anova_target = sys.argv[12]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
if anova_target == "hgs_delta":
    plot_target = ["hgs_delta", "hgs_corrected_delta"]

elif anova_target == "hgs_predicted":
    plot_target = ["hgs_predicted", "hgs_corrected_predicted"]

elif anova_target == "hgs":
    plot_target = ["hgs"]

##############################################################################

df = pd.DataFrame()
for target in ["hgs_left", "hgs_right", "hgs_L+R"]:
    df_data, df_anova_result, df_post_hoc_result_without_gender, df_post_hoc_result_with_gender =  load_disorder_anova_results(
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
    
    df = pd.concat([df, df_data], axis=0)
    
df_pre = df[df["disorder_episode"].str.startswith("pre")]
df_post = df[df["disorder_episode"].str.startswith("post")]

df_pre_disorder = df_pre[df_pre['treatment']==f"{population}"]
df_post_disorder = df_post[df_post['treatment']==f"{population}"]

df_pre_control = df_pre[df_pre['treatment']=="control"]
df_post_control = df_post[df_post['treatment']=="control"]

print("===== Done! =====")
embed(globals(), locals())

