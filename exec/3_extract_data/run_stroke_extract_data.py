import sys
import os
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.extract_features import stroke_extract_features
from hgsprediction.extract_target import stroke_extract_target
from hgsprediction.save_data import stroke_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
gender = sys.argv[7]

if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"

df = stroke_load_data.load_computed_targets_data(population, mri_status, session_column)

if gender == "both_gender":
    df = df
elif gender == "female":
    df = df[df["31-0.0"] == 0.0]
elif gender == "male":
    df = df[df["31-0.0"] == 1.0]


feature_extractor = stroke_extract_features.StrokeExtractFeatures(df, mri_status, stroke_cohort, visit_session, feature_type)
target_extractor = stroke_extract_target.StrokeExtractTarget(df, mri_status, stroke_cohort, visit_session, target)

extracted_features, feature_list = feature_extractor.extract_features(df)
extracted_target, target_list = target_extractor.extract_target(df)

stroke_save_data.save_extracted_features_data(extracted_features, population, mri_status, session_column, feature_type, gender)
stroke_save_data.save_extracted_target_data(extracted_target, population, mri_status, session_column, target, gender)

print("===== Done! =====")
embed(globals(), locals())
