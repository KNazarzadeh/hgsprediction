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


###############################################################################
jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    "GMV",
)

jay_file_1 = os.path.join(
        jay_path,
        '1_gmd_schaefer_all_subjects.jay')
jay_file_2 = os.path.join(
        jay_path,
        '2_gmd_SUIT_all_subjects.jay')
jay_file_4 = os.path.join(
        jay_path,
        '4_gmd_tian_all_subjects.jay')


feature_dt_1 = dt.fread(jay_file_1)
feature_df_1 = feature_dt_1.to_pandas()
feature_df_1.set_index('SubjectID', inplace=True)

feature_dt_2 = dt.fread(jay_file_2)
feature_df_2 = feature_dt_2.to_pandas()
feature_df_2.set_index('SubjectID', inplace=True)

feature_dt_4 = dt.fread(jay_file_4)
feature_df_4 = feature_dt_4.to_pandas()
feature_df_4.set_index('SubjectID', inplace=True)

gm_anthro_all = pd.concat([feature_df_1, feature_df_2, feature_df_4], axis=1)
gm_anthro_all = gm_anthro_all.dropna()


gm_anthro_all.index = gm_anthro_all.index.str.replace("sub-", "")
gm_anthro_all.index = gm_anthro_all.index.map(int)
##############################################################################
# load data
df_combined = pd.DataFrame()
df_1st_scan = healthy.load_hgs_predicted_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session="2",
)

df_2nd_scan = healthy.load_hgs_predicted_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session="3",
)

# Find the intersection of indexes
intersection_index = df_1st_scan.index.intersection(df_2nd_scan.index).intersection(gm_anthro_all.index)

df_1st_scan_intersected = df_1st_scan[df_1st_scan.index.isin(intersection_index)]
df_2nd_scan_intersected = df_2nd_scan[df_2nd_scan.index.isin(intersection_index)]
gm_anthro_overlap = gm_anthro_all[gm_anthro_all.index.isin(intersection_index)]

intersection_index_female = df_1st_scan_intersected[df_1st_scan_intersected["gender"]==0].index.intersection(df_2nd_scan_intersected[df_2nd_scan_intersected["gender"]==0].index)
intersection_index_male = df_1st_scan_intersected[df_1st_scan_intersected["gender"]==1].index.intersection(df_2nd_scan_intersected[df_2nd_scan_intersected["gender"]==1].index)

df_1st_scan_intersected_female = df_1st_scan_intersected[df_1st_scan_intersected.index.isin(intersection_index_female)]
df_1st_scan_intersected_male = df_1st_scan_intersected[df_1st_scan_intersected.index.isin(intersection_index_male)]
df_2nd_scan_intersected_female = df_2nd_scan_intersected[df_2nd_scan_intersected.index.isin(intersection_index_female)]
df_2nd_scan_intersected_male = df_2nd_scan_intersected[df_2nd_scan_intersected.index.isin(intersection_index_male)]

gm_anthro_overlap_female = gm_anthro_overlap[gm_anthro_overlap.index.isin(intersection_index_female)]
gm_anthro_overlap_male = gm_anthro_overlap[gm_anthro_overlap.index.isin(intersection_index_male)]


print("===== Done! =====")
embed(globals(), locals())

##############################################################################
n_regions = gm_anthro_overlap.shape[1]
y_axis = ["actual", "predicted", "actual-predicted"]
x_axis = gm_anthro_overlap.columns.tolist()[:n_regions]

df_corr_1st_scan, df_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(gm_anthro_overlap, df_1st_scan_intersected, y_axis, x_axis)
df_female_corr_1st_scan, df_female_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(gm_anthro_overlap_female, df_1st_scan_intersected_female, y_axis, x_axis)
df_male_corr_1st_scan, df_male_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(gm_anthro_overlap_male, df_1st_scan_intersected_male, y_axis, x_axis)
print(df_corr_1st_scan)
print(df_female_corr_1st_scan)
print(df_male_corr_1st_scan)

df_corr_2nd_scan, df_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(gm_anthro_overlap, df_2nd_scan_intersected, y_axis, x_axis)
df_female_corr_2nd_scan, df_female_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(gm_anthro_overlap_female, df_2nd_scan_intersected_female, y_axis, x_axis)
df_male_corr_2nd_scan, df_male_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(gm_anthro_overlap_male, df_2nd_scan_intersected_male, y_axis, x_axis)
print(df_corr_2nd_scan)
print(df_female_corr_2nd_scan)
print(df_male_corr_2nd_scan)


