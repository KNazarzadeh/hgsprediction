import sys
import os
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.old_define_features import stroke_define_features
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

if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"

df_original = stroke_load_data.load_computed_targets_data(population, mri_status, session_column)

for gender in ["female", "male"]:
    if gender == "female":
        df = df_original[df_original["31-0.0"] == 0.0]
    if gender == "male":
        df = df_original[df_original["31-0.0"] == 1.0]

    feature_extractor = stroke_define_features.StrokeExtractFeatures(df, mri_status, stroke_cohort, visit_session, feature_type)
    target_extractor = stroke_extract_target.StrokeExtractTarget(df, mri_status, stroke_cohort, visit_session, target)

    feature_list = feature_extractor.extract_features()
    extracted_features = df[feature_list]
    target_list = target_extractor.extract_target()
    extracted_target = df[target_list]

    extracted_data = pd.concat([extracted_features, extracted_target], axis=1)

    extracted_data = extracted_data.dropna()
    
    stroke_save_data.save_extracted_data(extracted_data, population, mri_status, session_column, feature_type, target, gender)


print("===== Done! =====")
embed(globals(), locals())
