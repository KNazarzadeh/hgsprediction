

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hgsprediction.load_data import load_prepared_data
from ptpython.repl import embed
from hgsprediction.save_plot import save_plots

###############################################################################
filename = sys.argv[0]
motor = sys.argv[1]
population = sys.argv[2]
mri_status = sys.argv[3]

df = load_prepared_data(df_name="available_hgs",
                   motor=motor,
                   population=population,
                   mri_status=mri_status)

df_pre = load_prepared_data(df_name="pre_disease",
                   motor=motor,
                   population=population,
                   mri_status=mri_status)

df_post = load_prepared_data(df_name="post_disease",
                   motor=motor,
                   population=population,
                   mri_status=mri_status)

df_longitudinal = load_prepared_data(df_name="longitudinal_disease",
                   motor=motor,
                   population=population,
                   mri_status=mri_status)

save_path = save_plots(plot_name="display_summary_disease",
                        motor=motor,
                        population=population,
                        mri_status=mri_status)


fig, ax = plt.subplots(figsize=(80, 50))

pre_df_len = len(df_pre)

# mask = (df_pre['46-2.0'].isna()) & (df_pre['47-2.0'].isna())
# df_pre['followup_days-2.0'][mask] = np.NaN
    
df_all = pd.concat([df_pre, df_post, df_longitudinal])

filter_col = [col for col in df if col.startswith('followup_days')]
sub_df_days = pd.DataFrame(df_all, columns=filter_col)

color_range = ['g', 'deeppink', 'blue', 'chocolate']

for indx in range(0, len(sub_df_days)):
    ar = ~sub_df_days.iloc[indx, 0:].isna().values
    sessions = [i for i, val in enumerate(ar) if val] 
    x = sub_df_days.iloc[indx, 0:sessions[-1]+1]
    x = x.dropna()
    y = indx+2
    plt.plot(x, np.zeros_like(x)+y, '-o', linewidth=6, markersize=30, 
        markerfacecolor='none', color='lightgrey')
    for ses in range(0, len(sessions)):      
        x = sub_df_days.iloc[indx, sessions[ses]]
        y= indx+2
        plt.plot(x, y, 'o', markerfacecolor=color_range[sessions[ses]], markersize=30)

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(12)
ax.spines['bottom'].set_linewidth(12)

plt.axvline(x=180, ymin=0, ymax=1, color='red', linestyle='--', linewidth=12)
plt.axvline(x=365, ymin=0, ymax=1, color='red', linestyle='--', linewidth=12)
plt.axvline(x=730, ymin=0, ymax=1, color='red', linestyle='--', linewidth=12)

plt.xticks(np.arange(sub_df_days.min().min(), sub_df_days.max().max(), step=200),
    rotation=90, fontsize=60)
plt.yticks(color='w')
# plt.legend(['g', 'deeppink', 'blue', 'chocolate'],["Session 0", "Session 1", "Session 2", "Session 3"], loc='upper right')

plt.show()
fig.savefig(f"{mri_status}.png")
plt.close()

