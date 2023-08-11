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

# from hgsprediction.plots import create_regplot

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
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
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
df_female["days"] = df_female[df_female['31-0.0']==0.0]['post_days']
df_male["days"] = df_male[df_male['31-0.0']==1.0]['post_days']
##############################################################################
#female
y_true = df_female[y]
y_pred = female_best_model_trained.predict(df_female[X])
df_female["hgs_actual"] = y_true
df_female["hgs_predicted"] = y_pred
df_female["hgs_acutal-predicted"] = y_true - y_pred
# corr_female_diff, p_female_diff = spearmanr(df_female["hgs_diff"], f_days/365)

#male
y_true = df_male[y]
y_pred = male_best_model_trained.predict(df_male[X])
df_male["hgs_actual"] = y_true
df_male["hgs_predicted"] = y_pred
df_male["hgs_acutal-predicted"] = y_true - y_pred

df_female_output = pd.concat([df_female[X], df_female[y]], axis=1)
df_female_output = pd.concat([df_female_output, df_female[['hgs_predicted', 'hgs_acutal-predicted']]], axis=1)

df_male_output = pd.concat([df_male[X], df_male[y]], axis=1)
df_male_output = pd.concat([df_male_output, df_male[['hgs_predicted', 'hgs_acutal-predicted']]], axis=1)

print(df_female_output)
print(df_male_output)

df_both_gender = pd.concat([df_female, df_male], axis=0)
print(df_both_gender)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################################################################################################
# create_regplot(df=df_both_gender,
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
