import sys
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.compute_features import stroke_compute_features
from hgsprediction.save_data import stroke_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
# stroke_cohort = sys.argv[4]
# visit_session = sys.argv[5]
# gender = sys.argv[6]


# if visit_session == "1":
#     session_column = f"1st_{stroke_cohort}_session"
# elif visit_session == "2":
#     session_column = f"2nd_{stroke_cohort}_session"
# elif visit_session == "3":
#     session_column = f"3rd_{stroke_cohort}_session"
# elif visit_session == "4":
#     session_column = f"4th_{stroke_cohort}_session"

# df = stroke_load_data.load_validated_hgs_data(population, mri_status, session_column, gender="both_gender")

# df = stroke_compute_features.compute_features(df, session_column, feature_type)

# df_female = df[df["31-0.0"]==0.0]
# df_male = df[df["31-0.0"]==1.0]

# stroke_save_data.save_preprocessed_data(df, population, mri_status, session_column, "both_gender")
# stroke_save_data.save_preprocessed_data(df_female, population, mri_status, session_column, "female")
# stroke_save_data.save_preprocessed_data(df_male, population, mri_status, session_column, "male")

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
# for all session pre- and -post stroke together (all in one):
# for stroke_cohort in ["pre-stroke", "post-stroke", "longitudinal-stroke"]:
for stroke_cohort in ["pre-stroke", "post-stroke"]:
    if stroke_cohort == "longitudinal-stroke":
        for visit_session in range(1, 2):
            if visit_session == 1:
                session_column = f"1st_{stroke_cohort}_session"
            df = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, stroke_cohort)
            for stroke_subgroup in ["pre-stroke", "post-stroke"]:
                if visit_session == 1:
                    subgroup_session_column = f"1st_{stroke_subgroup}_session"
                df = stroke_compute_features.compute_features(df, subgroup_session_column, feature_type, mri_status)
                print(df)
            stroke_save_data.save_preprocessed_data(df, population, mri_status, session_column, stroke_cohort)
            print(stroke_cohort)
    else:
        for visit_session in visit_range:
            if visit_session == 1:
                session_column = f"1st_{stroke_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{stroke_cohort}_session"
            elif visit_session == 3:
                session_column = f"3rd_{stroke_cohort}_session"
            df = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, stroke_cohort)
            df = stroke_compute_features.compute_features(df, session_column, feature_type, mri_status)
        
            stroke_save_data.save_preprocessed_data(df, population, mri_status, session_column, stroke_cohort)
            print(stroke_cohort)
            print(df)

print("===== Done! =====")
embed(globals(), locals())