import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.input_arguments import parse_args, input_arguments
from hgsprediction.load_imaging_data import load_imaging_data
from hgsprediction.load_trained_model import load_best_model_trained
from hgsprediction.prepare_stroke.prepare_stroke_data import prepare_stroke
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from hgsprediction.plots import create_regplot

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


filename = sys.argv[0]
population = sys.argv[1]

img_type = sys.argv[1]
neuroanatomy = sys.argv[2]

###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model_name, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)
    
###############################################################################
female_best_model_trained = load_best_model_trained(
                                population,
                                "female",
                                feature_type,
                                target,
                                confound_status,
                                model_name,
                                cv_repeats_number,
                                cv_folds_number,
                            )
print(female_best_model_trained)

male_best_model_trained = load_best_model_trained(
                                population,
                                "male",
                                feature_type,
                                target,
                                confound_status,
                                model_name,
                                cv_repeats_number,
                                cv_folds_number,
                            )
print(male_best_model_trained)
##############################################################################
stroke_all_columns, df_female, df_male, X, y = prepare_stroke(target)

##############################################################################
f_days = df_female[df_female['31-0.0']==0.0]['post_days']
m_days = df_male[df_male['31-0.0']==1.0]['post_days']

##############################################################################
#female
y_true = df_female[y]
y_pred = female_best_model_trained.predict(df_female[X])
df_female["actual_hgs"] = y_true
df_female["predicted_hgs"] = y_pred
df_female["hgs_diff"] = y_true - y_pred
# corr_female_diff, p_female_diff = spearmanr(df_female["hgs_diff"], f_days/365)

#male
y_true = df_male[y]
y_pred = male_best_model_trained.predict(df_male[X])
df_male["actual_hgs"] = y_true
df_male["predicted_hgs"] = y_pred
df_male["hgs_diff"] = y_true - y_pred
# corr_male, p_male = spearmanr(df_male["hgs_diff"], m_days/365)
# print("===== Done! =====")
# embed(globals(), locals())
df_female_output = pd.concat([df_female[X], df_female[y]], axis=1)
df_female_output = pd.concat([df_female_output, df_female[['predicted_hgs', 'hgs_diff']]], axis=1)

df_male_output = pd.concat([df_male[X], df_male[y]], axis=1)
df_male_output = pd.concat([df_male_output, df_male[['predicted_hgs', 'hgs_diff']]], axis=1)
print(df_female_output)
###############################################################################################################################################################
create_regplot(target,
            x_male=m_days/365,
            y_male=df_male["predicted_hgs"],
            x_female=f_days/365,
            y_female=df_female["predicted_hgs"],
            corr_male=spearmanr(df_male["predicted_hgs"], m_days/365)[0],
            corr_female=spearmanr(df_female["predicted_hgs"], f_days/365)[0],
            title="Predicted",
            y_label="Predicted HGS",)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################################################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
fig = plt.figure(figsize=(15,15))
sns.set_context("poster")
ax = fig.add_subplot(111)
plt.tick_params(axis='both', labelsize=30)
sns.regplot(x=m_days/365, y=df_male["hgs_diff"], line_kws={"color": "grey"}, scatter_kws={"color": "blue"})
sns.regplot(x=f_days/365, y=df_female["hgs_diff"], scatter_kws={"color": "red"}, line_kws={"color": "grey"})

ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# Add two text annotations to the corner
text_female = 'r= ' + str(format(spearmanr(df_female["predicted_hgs"], f_days)[0], '.3f'))
text_male = 'r= ' + str(format(spearmanr(df_male["predicted_hgs"], m_days)[0], '.3f'))

xmax = np.max(ax.get_xlim())
ymax = np.max(ax.get_ylim())
xmin = np.min(ax.get_xlim())
ymin = np.min(ax.get_ylim())

plt.text(xmax - 0.15 * xmax, ymax - 0.005 * ymax,text_female, verticalalignment='top',
         horizontalalignment='right', fontsize=24, fontweight="bold", color="red")

plt.text(xmax - 0.15 * xmax, ymax - 0.07 * ymax, text_male, verticalalignment='top',
         horizontalalignment='right', fontsize=24, fontweight="bold", color="blue")

plt.xlabel('Post-stroke years', fontsize=30, fontweight="bold")
plt.ylabel('Predicted HGS', fontsize=30, fontweight="bold")
plt.title("Difference (Actual-Predicted)HGS vs Post-stroke years", fontsize=30, fontweight="bold", y=1.05)
plt.tight_layout()  # Adjust layout to prevent clipping of legend

ax_lim_min = np.min(plt.xlim())
ax_lim_max = np.max(plt.xlim())

plt.plot(.5,.5, 'ko')

plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')

plt.show()
plt.savefig(f"090823correlate_diff_actual_predicted_hgs_storke_mri_both_gender_post_diff_years.png")
plt.close()
