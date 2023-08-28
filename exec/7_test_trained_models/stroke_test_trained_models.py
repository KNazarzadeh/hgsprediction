import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.input_arguments import parse_args, input_arguments
from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.extract_data import stroke_extract_data
# from hgsprediction.plots import plot_correlation_hgs

from hgsprediction.prepare_stroke.prepare_stroke_data import prepare_stroke
from hgsprediction.old_define_features import stroke_define_features
from hgsprediction.extract_target import stroke_extract_target
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import load_trained_models


from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# from hgsprediction.plots import create_regplot

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]

if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"

###############################################################################

female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "female",
                                feature_type,
                                target,
                                "linear_svm",
                                10,
                                5,
                            )

print(female_best_model_trained)

male_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "male",
                                feature_type,
                                target,
                                "linear_svm",
                                10,
                                5,
                            )
print(male_best_model_trained)
##############################################################################
# load data
df_female = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, "female")
df_female = df_female[(df_female["1st_post-stroke_session"]==2.0) | (df_female["1st_post-stroke_session"]== 3.0)]
df_male = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, "male")
df_male = df_male[(df_male["1st_post-stroke_session"]==2.0) | (df_male["1st_post-stroke_session"]== 3.0)]

features = define_features(feature_type)

data_extracted_female = stroke_extract_data.extract_data(df_female, stroke_cohort, visit_session, features, target)
data_extracted_male = stroke_extract_data.extract_data(df_male, stroke_cohort, visit_session, features, target)

X = features
y = target

##############################################################################
days_female = df_female.loc[data_extracted_female.index, "1st_post-stroke_days"]

days_male = df_male.loc[data_extracted_male.index, "1st_post-stroke_days"]

data_extracted_female = pd.concat([data_extracted_female, days_female], axis=1)
data_extracted_male = pd.concat([data_extracted_male, days_male], axis=1)

##############################################################################
#female
y_true = data_extracted_female[y]
y_pred = female_best_model_trained.predict(data_extracted_female[X])
data_extracted_female["hgs_actual"] = y_true
data_extracted_female["hgs_predicted"] = y_pred
data_extracted_female["hgs_actual-predicted"] = y_true - y_pred
data_extracted_female.loc[:, "gender"] = 0
data_extracted_female.loc[:, "years"] = data_extracted_female.loc[:, "1st_post-stroke_days"]/365

# corr_female_diff, p_female_diff = spearmanr(df_female["hgs_diff"], f_days/365)

#male
y_true = data_extracted_male[y]
y_pred = male_best_model_trained.predict(data_extracted_male[X])
data_extracted_male["hgs_actual"] = y_true
data_extracted_male["hgs_predicted"] = y_pred
data_extracted_male["hgs_actual-predicted"] = y_true - y_pred
data_extracted_male.loc[:, "gender"] = 1
data_extracted_male.loc[:, "years"] = data_extracted_male.loc[:, "1st_post-stroke_days"]/365


df_both_gender = pd.concat([data_extracted_female, data_extracted_male], axis=0)
print(df_both_gender)

# Create the actual HGS vs predicted HGS plot for females and fefemales separately
def plot_correlation(data, x, y, x_label, y_label, target):
    
    fig, ax = plt.subplots(figsize=(20,10))
    sns.set_context("poster")
    ax.set_box_aspect(1)
    sns.regplot(data=data, x=x, y=y, ax=ax, line_kws={"color": "grey"}, scatter=False)
    sns.scatterplot(data=data, x=x, y=y, hue="gender", palette=['red', 'blue'])

    ax.tick_params(axis='both', labelsize=20)

    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    
    text = 'r = ' + str(format(spearmanr(data[y], data[x])[0], '.3f'))
    
    ax.set_xlabel(f"{x_label}", fontsize=20, fontweight="bold")
    ax.set_ylabel(f"{y_label} HGS", fontsize=20, fontweight="bold")

    ax.set_title(f"{y_label} HGS vs {x_label} - Target={target} (Females={len(data_extracted_female)}, Males={len(data_extracted_male)})", fontsize=15, fontweight="bold", y=1)
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=18, fontweight="bold")
    legend = ax.legend(title="Gender", loc="lower right")
    # Modify individual legend labels
    new_legend_labels = ['Female', 'Male']
    for text, label in zip(legend.get_texts(), new_legend_labels):
        text.set_text(label)
    # Plot regression line    
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')

    plt.show()
    plt.savefig(f"MRI_records_{y_label} vs {x_label} - {target}.png")
    plt.close()

def corr_score(data, x, y):
    corr, p_val = spearmanr(data[y], data[x])
    return corr, data[[y,x]]

corr, df_corr = corr_score(data=df_both_gender,
           x="years",
           y="hgs_actual-predicted")
print(df_corr)
print(corr)

plot_correlation(data=df_both_gender,
                 x="years",
                 y="hgs_actual-predicted",
                 x_label="Post-stroke years",
                 y_label="(Actual-Predicted)",
                 target=target.replace('_', ' ').upper())

print("===== Done! =====")
embed(globals(), locals())
###############################################################################################################################################################
# plot_correlation_hgs(df=df_both_gender,
#                 x="years",
#                 y="predicted",
#                 title="Predicted",
#                 y_label="Predicted",
#                 target,
#                 feature_type,
#                 stroke_type,
#                 gender,
# )
print("===== Done! =====")
embed(globals(), locals())
###############################################################################################################################################################
