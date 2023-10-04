
#%%
import sys
import numpy as np
import pandas as pd
from psmpy import PsmPy
from hgsprediction.load_results import healthy
from hgsprediction.load_results import stroke
import matplotlib.pyplot as plt
import seaborn as sns

from ptpython.repl import embed
# # print("===== Done! =====")
# # embed(globals(), locals())

#%%
def match(data, target, m, seed=None):
    '''
    Generates a matched sample of size m from data.
    data: pandas dataframe
    target: name of the target variable, must be binary (string)
    confound: list of confounding variables to match on (strings)
    m : number of samples to draw
    seed : random seed
    
    Returns: index of matched sample
    '''
    if seed is not None:
        np.random.seed(seed)
    # randomly select a target
    targets = np.unique(data[target])
    assert(len(targets) == 2) # Currently only two targets supported.
    atarget = np.random.choice(targets)
    # make sure that we can sample from this target    
    n_samp = round(m/2)
    idx = data[target] == atarget
    idx2 = ~idx
    assert(n_samp < np.sum(idx))    
    idx = np.random.choice(np.where(idx)[0], n_samp, replace=False)    
    # combine idx and idx2
    idx = np.concatenate((idx, np.where(idx2)[0]))
    print(len(idx))
    # prepare data for PsmPy    
    col_for_psm = [target]
    print(col_for_psm)
    df = data[col_for_psm]    
    # retain only the sampled rows
    df = df.iloc[idx,:]
    df['index'] = df.index    

    psm = PsmPy(df, treatment=target, indx='index')
    psm.logistic_ps(balance=True)
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)

    index = psm.df_matched['index'].values
    
    return index

###############################################################################
#%%
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

df_mri_1st_scan = healthy.load_hgs_predicted_results("healthy",
    "mri",
    "linear_svm",
    "anthropometrics_age",
    "hgs_L+R",
    "both_gender",
    session="2",
)
df_healthy = df_mri_1st_scan[["gender", "1st_scan_handedness", "1st_scan_age", "1st_scan_bmi",  "1st_scan_height",  "1st_scan_waist_to_hip_ratio", f"1st_scan_{target}"]]

df_healthy.rename(columns={"1st_scan_age":"age", "1st_scan_bmi":"bmi",  "1st_scan_height":"height",  "1st_scan_waist_to_hip_ratio":"waist_to_hip_ratio",
                           "1st_scan_handedness":"handedness", f"1st_scan_{target}":f"{target}"}, inplace=True)
print(df_healthy)

df_healthy_female = df_mri_1st_scan[df_mri_1st_scan['gender']==0]
df_healthy_male = df_mri_1st_scan[df_mri_1st_scan['gender']==1]

###############################################################################
#%%
stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_stroke = stroke.load_hgs_predicted_results("stroke", mri_status, session_column, model_name, feature_type, target, "both_gender")
df_pre_stroke = df_stroke[["gender", "1st_pre-stroke_handedness", "1st_pre-stroke_age", "1st_pre-stroke_bmi",  "1st_pre-stroke_height",  "1st_pre-stroke_waist_to_hip_ratio", f"1st_pre-stroke_{target}"]]
df_pre_stroke.rename(columns={"1st_pre-stroke_age":"age", "1st_pre-stroke_bmi":"bmi",  "1st_pre-stroke_height":"height",  "1st_pre-stroke_waist_to_hip_ratio":"waist_to_hip_ratio", 
                              "1st_pre-stroke_handedness":"handedness", f"1st_pre-stroke_{target}":f"{target}"}, inplace=True)

df_post_stroke = df_stroke[["gender", "1st_post-stroke_handedness", "1st_post-stroke_age", "1st_post-stroke_bmi",  "1st_post-stroke_height",  "1st_post-stroke_waist_to_hip_ratio", f"1st_post-stroke_{target}"]]
df_post_stroke.rename(columns={"1st_post-stroke_age":"age", "1st_post-stroke_bmi":"bmi",  "1st_post-stroke_height":"height",  "1st_post-stroke_waist_to_hip_ratio":"waist_to_hip_ratio",
                               "1st_post-stroke_handedness":"handedness", f"1st_post-stroke_{target}":f"{target}"}, inplace=True)

df_stroke_female = df_stroke[df_stroke['gender']==0]
df_stroke_male = df_stroke[df_stroke['gender']==1]

###############################################################################
#%%
df_pre_stroke["disease"] = 0
df_healthy["disease"] = 1

df_pre = pd.concat([df_healthy, df_pre_stroke])
df_pre['index'] = df_pre.index

df_post_stroke["disease"] = 0
df_healthy["disease"] = 1
df_post = pd.concat([df_healthy, df_post_stroke])
df_post['index'] = df_post.index

df_pre_female=df_pre[df_pre["gender"]==0]
df_pre_male=df_pre[df_pre["gender"]==1]
df_post_female=df_post[df_post["gender"]==0]
df_post_male=df_post[df_post["gender"]==1]
###############################################################################
psm = PsmPy(df_pre_female, treatment='disease', indx='index', exclude=["gender","handedness", "bmi",  "height",  "waist_to_hip_ratio",  "hgs_L+R"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
# psm.knn_matched_12n(matcher='propensity_logit', how_many=10)
df_pre_matched_female = psm.df_matched
# ax[0,0] = sns.barplot(df_pre_matched_female, x='propensity_logit', hue='disease')
# ax[0,0] = psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity logit', names = ['disease', 'control'], colors=['#E69F00', '#56B4E9'] ,save=True)
print("===== Done! =====")
embed(globals(), locals())
df_pre_matched_female_controls = psm.matched_ids["matched_ID"]
df_pre_matched_female_stroke = psm.matched_ids["index"]

df_healthy_pre_female = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_pre_matched_female_controls)]

###############################################################################

psm = PsmPy(df_pre_male, treatment='disease', indx='index', exclude=["gender","handedness", "hgs_L+R"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
# psm.knn_matched_12n(matcher='propensity_logit', how_many=10)
df_pre_matched_male = psm.df_matched
# ax[1,0] = sns.barplot(df_pre_matched_male, x='propensity_logit', hue='disease')
# psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity logit', names = ['disease', 'control'], colors=['#E69F00', '#56B4E9'] ,save=True)

df_pre_matched_male_controls = psm.matched_ids["matched_ID"]
df_pre_matched_male_stroke = psm.matched_ids["index"]
df_healthy_pre_male = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_pre_matched_male_controls)]
###############################################################################
psm = PsmPy(df_post_female, treatment='disease', indx='index', exclude=["gender","handedness", "hgs_L+R"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
# psm.knn_matched_12n(matcher='propensity_logit', how_many=10)
df_post_matched_female = psm.df_matched
# ax[0,1] = sns.barplot(df_post_matched_female, x='propensity_logit', hue='disease')
# psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity logit', names = ['disease', 'control'], colors=['#E69F00', '#56B4E9'] ,save=False)

df_post_matched_female_controls = psm.matched_ids["matched_ID"]
df_post_matched_female_stroke = psm.matched_ids["index"]
df_healthy_post_female = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_post_matched_female_controls)]

###############################################################################
psm = PsmPy(df_post_male, treatment='disease', indx='index', exclude=["gender","handedness", "hgs_L+R"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
# psm.knn_matched_12n(matcher='propensity_logit', how_many=10)
df_post_matched_male = psm.df_matched
# ax[1,1] = sns.barplot(df_post_matched_male, x='propensity_logit', hue='disease')
# psm.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel= 'Propensity logit', names = ['disease', 'control'], colors=['#E69F00', '#56B4E9'] ,save=False)

df_post_matched_male_controls = psm.matched_ids["matched_ID"]
df_post_matched_male_stroke = psm.matched_ids["index"]
df_healthy_post_male = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_post_matched_male_controls)]

fig, ax = plt.subplots(2,2, figsize=(16,10))

sns.histplot(df_pre_matched_female, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[0,0])
sns.histplot(df_post_matched_female, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[0,1])
sns.histplot(df_pre_matched_male, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[1,0])
sns.histplot(df_post_matched_male, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[1,1])

# Add titles to the subplots
ax[0, 0].set_title(f'Pre-Matched Female(N={len(df_pre_matched_female[df_pre_matched_female["disease"]==1])})')
ax[0, 1].set_title(f'Post-Matched Female(N={len(df_post_matched_female[df_post_matched_female["disease"]==1])})')
ax[1, 0].set_title(f'Pre-Matched Male(N={len(df_pre_matched_male[df_pre_matched_male["disease"]==1])})')
ax[1, 1].set_title(f'Post-Matched Male(N={len(df_post_matched_male[df_post_matched_male["disease"]==1])})')

# # Add custom legend labels
legend_labels = ['healthy', 'stroke']  # Replace with your desired labels

# Create custom legends for each subplot
for i in range(2):
    for j in range(2):
        ax[i, j].legend(labels=legend_labels)


plt.suptitle("Pre- and Post-stroke and healthy matched samples")
# Adjust layout to prevent title overlap
plt.tight_layout()

plt.show()
plt.savefig("matching.png")

print("===== Done! =====")
embed(globals(), locals())
