import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.extract_data import stroke_extract_data
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import save_spearman_correlation_results
from hgsprediction.save_results import save_hgs_predicted_results
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import load_trained_models


from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# from hgsprediction.plots import plot_correlations
from hgsprediction.save_plot.save_correlations_plot import stroke_save_correlations_plot


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
model_name = sys.argv[7]

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
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
df_longitudinal = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, stroke_cohort)
df_longitudinal = df_longitudinal[df_longitudinal["1st_pre-stroke_height"]>= df_longitudinal["1st_post-stroke_height"]]

features = define_features(feature_type)
X = features
y = target

df_merged = pd.DataFrame()
df_extracted_pre_stroke = df_longitudinal[[col for col in df_longitudinal.columns if "pre-stroke" in col]]
df_extracted_post_stroke = df_longitudinal[[col for col in df_longitudinal.columns if "post-stroke" in col]]

df_extracted_pre_stroke = df_extracted_pre_stroke[[col for col in df_extracted_pre_stroke.columns if any(item in col for item in ['bmi', 'height', 'waist_to_hip_ratio', 'gender', y])]]
df_extracted_post_stroke = df_extracted_post_stroke[[col for col in df_extracted_post_stroke.columns if "age" in col]]

df_merged = pd.concat([df_extracted_pre_stroke, df_extracted_post_stroke], axis=1)

df_merged = df_merged.dropna(subset=[col for col in df_merged.columns if any(item in col for item in features)])
df_merged = df_merged[~df_merged.index.isin([2075055, 3869956, 4705336, 5894876])]
       
if visit_session == "1":
    prefix_pre = f"1st_pre-stroke_"
    prefix_post = f"1st_post-stroke_"
    df_merged.columns = df_merged.columns.str.replace(prefix_pre, "")
    df_merged.columns = df_merged.columns.str.replace(prefix_post, "")


df_female = df_merged[df_merged["gender"] == 0]
df_male = df_merged[df_merged["gender"] == 1]

df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
df_male = predict_hgs(df_male, X, y, male_best_model_trained, target) 

df_both_gender = pd.concat([df_female, df_male], axis=0)

##############################################################################
y_axis = ["actual", "predicted", "actual-predicted"]
x_axis = ["actual", "predicted"]
df_corr, df_pvalue = calculate_spearman_hgs_correlation(df_both_gender, y_axis, x_axis)
df_female_corr, df_female_pvalue = calculate_spearman_hgs_correlation(df_both_gender[df_both_gender["gender"]==0], y_axis, x_axis)
df_male_corr, df_male_pvalue = calculate_spearman_hgs_correlation(df_both_gender[df_both_gender["gender"]==1], y_axis, x_axis)

print(df_corr.applymap(lambda x: '{:.3f}'.format(x)))
print(df_female_corr.applymap(lambda x: '{:.3f}'.format(x)))
print(df_male_corr.applymap(lambda x: '{:.3f}'.format(x)))


print("===== Done! =====")
embed(globals(), locals())
