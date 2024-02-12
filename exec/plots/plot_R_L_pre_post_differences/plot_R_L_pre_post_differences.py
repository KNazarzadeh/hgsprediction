import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

cohort = "longitudinal-{population}"
session_column = f"1st_{cohort}_session"

longitudinal_mri = f"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/{population}/preprocessed_data/mri_{population}/longitudinal-{population}_data/1st_longitudinal-{population}_session_data/preprocessed_data/1st_longitudinal-{population}_session_preprocessed_data.csv"
longitudinal_nonmri = f"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/{population}/preprocessed_data/nonmri_{population}/longitudinal-{population}_data/1st_longitudinal-{population}_session_data/preprocessed_data/1st_longitudinal-{population}_session_preprocessed_data.csv"

df_longitudinal_mri = pd.read_csv(longitudinal_mri, sep=',', index_col=0)
df_longitudinal_nonmri = pd.read_csv(longitudinal_nonmri, sep=',', index_col=0)

df_long = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri])

longitudinal_folder = f"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/results_hgsprediction/{population}/mri+nonmri/1st_longitudinal-{population}_session/anthropometrics_age/hgs_L+R/linear_svm/hgs_predicted_results/both_gender_hgs_predicted_results.csv"
df_longitudinal = pd.read_csv(longitudinal_folder, sep=',', index_col=0)

df_long = df_long[df_long.index.isin(df_longitudinal.index)]

df_selected = pd.concat([df_long, df_longitudinal], axis=1)

df_selected["pre-post_hgs_right"] = df_selected[f"1st_pre-{population}_hgs_right"] - df_selected[f"1st_post-{population}_hgs_right"]
df_selected["pre-post_hgs_left"] = df_selected[f"1st_pre-{population}_hgs_left"] - df_selected[f"1st_post-{population}_hgs_left"]

df_selected['gender'] = df_selected['gender'].replace({0: 'female', 1: 'male'})

###############################################################################
###############################################################################

custom_palette = {'female': 'red', 'male': '#069AF3'}  # You can use any hex color codes you prefer

fig = plt.figure(figsize=(8,8))  # Adjust the figure size if needed
plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

ax = sns.scatterplot(x="pre-post_hgs_left", y="pre-post_hgs_right", data=df_selected, hue ="gender", palette=custom_palette, marker="$\circ$", s=150)
sns.regplot(x="pre-post_hgs_left", y="pre-post_hgs_right", data=df_selected, scatter=False, color='darkgrey')

# Add labels and title
plt.xlabel("Left HGS", fontsize=12, fontweight="bold")
plt.ylabel("Right HGS", fontsize=12, fontweight="bold")

ax.set_xlim([-20, 32])
ax.set_ylim([-20, 32])

# Set ticks for x-axis
plt.xticks(ticks=np.arange(-20, 31+1, 5))

# Set ticks for y-axis
plt.yticks(ticks=np.arange(-20, 31+1, 5)) 

plt.title(f"Pre-Post HGS differences Right vs Left-{population.capitalize()}(N={len(df_selected)})", fontsize=15, fontweight="bold")

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='20', fontsize='18', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.show()
plt.savefig(f"scatter_pre-post_hgs_difference_{population}.png", bbox_inches='tight')
plt.close()

###############################################################################
###############################################################################

print("===== Done! =====")
embed(globals(), locals())

b = df_selected.sort_values(by="pre-post_hgs_left", ascending=False)
b = b.reset_index()

custom_palette = {'female': 'red', 'male': '#069AF3'}  # You can use any hex color codes you prefer

fig = plt.figure(figsize=(15,15))  # Adjust the figure size if needed
plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

# ax = sns.lineplot(x=b.index, y="pre-post_hgs_left", data=b, hue='gender', palette=custom_palette)
# # Add markers using a scatter plot
# sns.scatterplot(x=b.index, y="pre-post_hgs_left", data=b, hue='gender', palette=custom_palette, marker="$\circ$", s=100, ax=ax)
# Create a joint plot
g = sns.jointplot(x=b.index, y="pre-post_hgs_left", data=b, hue='gender', palette=custom_palette, marker="$\circ$", s=70, legend=False)

# Overlay lines between markers

sns.lineplot(x=b.index, y="pre-post_hgs_left", data=b, hue='gender', palette=custom_palette, ax=g.ax_joint, legend=False)
    
g.ax_marg_x.remove()
# Add labels and title
plt.xlabel("Subjects", fontsize=12, fontweight="bold")
plt.ylabel("Left HGS", fontsize=12, fontweight="bold")

# ax.set_xlim([-20, 32])
# ax.set_ylim([-20, 32])

# # Set ticks for x-axis
# plt.xticks(ticks=np.arange(-20, 31+1, 5))

# # Set ticks for y-axis
# plt.yticks(ticks=np.arange(-20, 31+1, 5)) 

plt.title(f"(Pre-Post) Left HGS-{population.capitalize()}(N={len(df_selected)})", fontsize=15, fontweight="bold")

# Place legend outside the plot
# plt.legend(title="Gender", title_fontsize='20', fontsize='18', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
plt.savefig(f"scatter_pre-post_hgs_difference_{population}_left.png", bbox_inches='tight')
plt.close()


fig = plt.figure(figsize=(15,15))  # Adjust the figure size if needed
plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

# ax = sns.lineplot(x=b.index, y="pre-post_hgs_left", data=b, hue='gender', palette=custom_palette)
# # Add markers using a scatter plot
# sns.scatterplot(x=b.index, y="pre-post_hgs_left", data=b, hue='gender', palette=custom_palette, marker="$\circ$", s=100, ax=ax)
# Create a joint plot
g = sns.jointplot(x=b.index, y="pre-post_hgs_right", data=b, hue='gender', palette=custom_palette, marker="$\circ$", s=70, legend=False)

# Overlay lines between markers

sns.lineplot(x=b.index, y="pre-post_hgs_right", data=b, hue='gender', palette=custom_palette, ax=g.ax_joint, legend=False)
    
g.ax_marg_x.remove()
# Move x-axis to the top
g.ax_joint.xaxis.set_ticks_position('top')
g.ax_joint.xaxis.set_label_position('top')
# Remove bottom line
g.ax_joint.spines['bottom'].set_visible(False)
# Add labels and title
plt.xlabel("Subjects", fontsize=12, fontweight="bold")
plt.ylabel("Right HGS", fontsize=12, fontweight="bold")

# ax.set_xlim([-20, 32])
# ax.set_ylim([-20, 32])

# # Set ticks for x-axis
# plt.xticks(ticks=np.arange(-20, 31+1, 5))

# # Set ticks for y-axis
# plt.yticks(ticks=np.arange(-20, 31+1, 5)) 

plt.title(f"(Pre-Post) Left HGS-{population.capitalize()}(N={len(df_selected)})", fontsize=15, fontweight="bold")

# Place legend outside the plot
# plt.legend(title="Gender", title_fontsize='20', fontsize='18', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
plt.savefig(f"scatter_pre-post_hgs_difference_{population}_right.png", bbox_inches='tight')
plt.close()

print("===== Done! =====")
embed(globals(), locals())