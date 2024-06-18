import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.define_features import define_features
from hgsprediction.extract_data import disorder_extract_data
from hgsprediction.load_data.disorder import load_disorder_data
from hgsprediction.save_results.disorder.save_disorder_extracted_data_by_feature_and_target import save_disorder_extracted_data_by_feature_and_target
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
disorder_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
gender = sys.argv[7]
first_event = sys.argv[8]
##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
if mri_status == "mri+nonmri":
    df_longitudinal_mri = load_disorder_data.load_preprocessed_data(population, "mri", session_column, disorder_cohort, first_event)
    df_longitudinal_nonmri = load_disorder_data.load_preprocessed_data(population, "nonmri", session_column, disorder_cohort, first_event)
    df_longitudinal = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri]).dropna(axis=1, how='all')
else:
    df_longitudinal = load_disorder_data.load_preprocessed_data(population, mri_status, session_column, disorder_cohort, first_event)
##############################################################################



first_date_in_patient_diagnosis = df_longitudinal[[col for col in df_longitudinal.columns if "41280" in col]]
first_diagnosis = df_longitudinal[[col for col in df_longitudinal.columns if "41270" in col]]

primary_date_in_patients_diagnosis = df_longitudinal[[col for col in df_longitudinal.columns if "41262" in col]]
primary_code_in_patients_diagnosis = df_longitudinal[[col for col in df_longitudinal.columns if "41202" in col]]


secondary_date_in_patients_diagnosis = df_longitudinal[[col for col in df_longitudinal.columns if "41204" in col]]
secondary_code_in_patients_diagnosis = df_longitudinal[[col for col in df_longitudinal.columns if "41202" in col]]

print("===== Done! =====")
embed(globals(), locals())