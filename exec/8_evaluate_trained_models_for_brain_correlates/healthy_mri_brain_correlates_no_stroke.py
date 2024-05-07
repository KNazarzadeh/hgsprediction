import os
import pandas as pd
import numpy as np
import sys
from hgsprediction.load_results import healthy
from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.load_imaging_data import load_imaging_data
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import datatable as dt
from hgsprediction.extract_data import stroke_extract_data
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results.healthy import save_spearman_correlation_results, \
                                               save_hgs_predicted_results
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation_on_brain_correlations
from hgsprediction.save_results import save_data_overlap_hgs_predicted_brain_correlations_results,\
                                       save_spearman_correlation_on_brain_correlations_results


# from hgsprediction.plots import create_regplot

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]
session = sys.argv[6]
brain_correlation_type = sys.argv[7]

###############################################################################

jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    f"{brain_correlation_type.upper()}",
)

schaefer_file = os.path.join(
    jay_path,
    f"{brain_correlation_type.upper()}_Schaefer400x7_Mean.jay")
feature_dt_schaefer = dt.fread(schaefer_file)
feature_df_schaefer = feature_dt_schaefer.to_pandas()
feature_df_schaefer.set_index('SubjectID', inplace=True)

tian_file = os.path.join(
    jay_path,
    f"{brain_correlation_type.upper()}_Tian_Mean.jay")
feature_dt_tian = dt.fread(tian_file)
feature_df_tian = feature_dt_tian.to_pandas()
feature_df_tian.set_index('SubjectID', inplace=True)

df_brain_correlation = pd.concat([feature_df_schaefer, feature_df_tian], axis=1)

if brain_correlation_type == "gmv":
    suit_file = os.path.join(
        jay_path,
        f"{brain_correlation_type.upper()}_SUIT_Mean.jay")
    feature_dt_suit = dt.fread(suit_file)
    feature_df_suit = feature_dt_suit.to_pandas()
    feature_df_suit.set_index('SubjectID', inplace=True)
    df_brain_correlation = pd.concat([df_brain_correlation, feature_df_suit], axis=1)

    
df_brain_correlation = df_brain_correlation.dropna()
df_brain_correlation.index = df_brain_correlation.index.str.replace("sub-", "")
df_brain_correlation.index = df_brain_correlation.index.map(int)

##############################################################################
# load data
df = healthy.load_hgs_predicted_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
)

# Find the intersection of indexes
intersection_index = df.index.intersection(df_brain_correlation.index)

df_intersected = df[df.index.isin(intersection_index)]
df_brain_correlation_overlap = df_brain_correlation[df_brain_correlation.index.isin(intersection_index)]

intersection_index_female = df_intersected[df_intersected["gender"]==0].index
intersection_index_male = df_intersected[df_intersected["gender"]==1].index

df_intersected_female = df_intersected[df_intersected.index.isin(intersection_index_female)]
df_intersected_male = df_intersected[df_intersected.index.isin(intersection_index_male)]

df_brain_correlation_overlap_female = df_brain_correlation_overlap[df_brain_correlation_overlap.index.isin(intersection_index_female)]
df_brain_correlation_overlap_male = df_brain_correlation_overlap[df_brain_correlation_overlap.index.isin(intersection_index_male)]


save_data_overlap_hgs_predicted_brain_correlations_results(
    df_brain_correlation_overlap,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
    brain_correlation_type,
)

save_data_overlap_hgs_predicted_brain_correlations_results(
    df_brain_correlation_overlap_female,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    brain_correlation_type,
)

save_data_overlap_hgs_predicted_brain_correlations_results(
    df_brain_correlation_overlap_male,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    brain_correlation_type,
)
##############################################################################
n_regions = df_brain_correlation_overlap.shape[1]
y_axis = ["actual", "predicted", "actual-predicted"]
x_axis = df_brain_correlation_overlap.columns.tolist()[:n_regions]

df_corr, df_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap, df_intersected, y_axis, x_axis)
df_female_corr, df_female_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_female, df_intersected_female, y_axis, x_axis)
df_male_corr, df_male_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_male, df_intersected_male, y_axis, x_axis)
print(df_corr)
print(df_female_corr)
print(df_male_corr)

save_spearman_correlation_on_brain_correlations_results(
    df_corr,
    df_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
    brain_correlation_type,
)
save_spearman_correlation_on_brain_correlations_results(
    df_female_corr,
    df_female_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    brain_correlation_type,
)
save_spearman_correlation_on_brain_correlations_results(
    df_male_corr,
    df_male_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    brain_correlation_type,
)

print("===== Done! =====")
embed(globals(), locals())



