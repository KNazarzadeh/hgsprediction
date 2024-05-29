import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.define_features import define_features
from hgsprediction.extract_data import disorder_extract_data
from hgsprediction.load_data import load_disorder_data
from hgsprediction.load_results.load_disorder_extracted_data_by_features import load_disorder_extracted_data_by_features
from hgsprediction.save_results.disorder.save_describe_disorder_extracted_data_by_features import save_describe_disorder_extracted_data_by_features
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
# Define main features and extra features:
features, extend_features = define_features(feature_type)
##############################################################################
# Define X as main features and y as target:
X = features
y = target

##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
###############################################################################

df = load_disorder_extracted_data_by_features(
    population,
    mri_status,
    session_column,
    feature_type,
    target,
    gender,
    first_event,
)

subgroup_columns_pre = [col for col in df.columns if f"pre-{population}" in col]

subgroup_columns_post = [col for col in df.columns if f"post-{population}" in col]

summary_stats_pre = df[subgroup_columns_pre].describe().apply(lambda x: round(x, 2))
summary_stats_post = df[subgroup_columns_post].describe().apply(lambda x: round(x, 2))


print("summary_stats_pre:\n", summary_stats_pre)
print("summary_stats_post:\n", summary_stats_post)

save_describe_disorder_extracted_data_by_features(
    summary_stats_pre,
    summary_stats_post,
    population,
    mri_status,
    session_column,
    feature_type,
    target,
    gender,
    first_event,
)

print("===== Done! =====")
embed(globals(), locals())
