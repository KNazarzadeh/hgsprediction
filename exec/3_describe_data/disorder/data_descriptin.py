import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.define_features import define_features
from hgsprediction.extract_data import disorder_extract_data
from hgsprediction.load_data import disorder_load_data
from hgsprediction.load_results.load_disorder_extracted_data_by_features import load_disorder_extracted_data_by_features
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
print("===== Done! =====")
embed(globals(), locals())
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    
    subgroup_columns = [col for col in df.columns if disorder_subgroup in col]
    
    print("===== Done! =====")
    embed(globals(), locals())
