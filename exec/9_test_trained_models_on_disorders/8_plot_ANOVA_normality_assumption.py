import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
first_event = sys.argv[12]
anova_target = sys.argv[13]
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
    first_event,
)
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)

df_female = df[df["gender"]=="female"]
df_male = df[df["gender"]=="male"]

df_control = df[df['group']=='control']
df_disorder = df[df['group']==f'{population}']

df_control_pre = df_control[df_control["time_point"] == "pre-control"]
df_control_post = df_control[df_control["time_point"] == "post-control"]

df_disorder_pre = df_disorder[df_disorder["time_point"] == f"pre-{population}"]
df_disorder_post = df_disorder[df_disorder["time_point"] == f"post-{population}"]

# print("===== Done End! =====")
# embed(globals(), locals())
##############################################################################
folder_path = os.path.join("plot_distribution_controls_patients", f"{population}", f"{first_event}", f"{target}", f"{n_samples}_matched", "comparison_pre_post_time_points")
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
##############################################################################
# Visualise new distribution
# Map each gender-group combination to a specific color
custom_palette = {
    'male': "darkblue",
    'female': "palevioletred",
}

fig, ax = plt.subplots(2,2, figsize=(12, 6))

# Control vs patients male Pre
sns.kdeplot(data=df_control_pre[df_control_pre["gender"] == "male"], 
            x=f"{anova_target}", fill=True, color="grey", ax=ax[0][0], label='control')

# Disorder male pre
sns.kdeplot(data=df_disorder_pre[df_disorder_pre["gender"] == "male"], 
            x=f"{anova_target}", fill=True, color=custom_palette['male'], ax=ax[0][0], label='patient')
# Add legend for male Pre plot
ax[0][0].legend()
##############################################################################
# Control vs patients male Post
sns.kdeplot(data=df_control_post[df_control_post["gender"] == "male"], 
            x=f"{anova_target}", fill=True, color="grey", ax=ax[1][0], label='control')

# Disorder male Post
sns.kdeplot(data=df_disorder_post[df_disorder_post["gender"] == "male"], 
            x=f"{anova_target}", fill=True, color=custom_palette['male'], ax=ax[1][0], label='patient')
# Add legend for male Pre plot
ax[1][0].legend()
##############################################################################
# Control vs patients female Pre
sns.kdeplot(data=df_control_pre[df_control_pre["gender"] == "female"], 
            x=f"{anova_target}", fill=True, color="grey", ax=ax[0][1], label='control')

# Disorder male pre
sns.kdeplot(data=df_disorder_pre[df_disorder_pre["gender"] == "female"], 
            x=f"{anova_target}", fill=True, color=custom_palette['female'], ax=ax[0][1], label='patient')
# Add legend for male Pre plot
ax[0][1].legend()
##############################################################################
# Control vs patients female Post
sns.kdeplot(data=df_control_post[df_control_post["gender"] == "female"], 
            x=f"{anova_target}", fill=True, color="grey", ax=ax[1][1], label='control')

# Disorder male Post
sns.kdeplot(data=df_disorder_post[df_disorder_post["gender"] == "female"], 
            x=f"{anova_target}", fill=True, color=custom_palette['female'], ax=ax[1][1], label='patient')
# Add legend for male Pre plot
ax[1][1].legend()
##############################################################################
ax[0][0].set_title('Males (Pre Time-point)', fontsize="14")
ax[1][0].set_title('Males (Post Time-point)', fontsize="14")

ax[0][1].set_title('Females (Pre Time-point)', fontsize="14")
ax[1][1].set_title('Females (Post Time-point)', fontsize="14")

# Add main title
fig.suptitle(f"{population.capitalize()}\nComparison of HGS between Matched controls and Patients at Pre and Post", fontsize=12, weight="bold")

plt.tight_layout()

plt.show()
file_path = os.path.join(folder_path, f"matched_sample_distribution_{population}_{anova_target}.png")
plt.savefig(file_path)
plt.close()

print("===== Done End! =====")
embed(globals(), locals())
###############################################################################


