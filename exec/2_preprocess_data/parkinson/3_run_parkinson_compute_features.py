import sys
import pandas as pd
from hgsprediction.load_data import parkinson_load_data
from hgsprediction.compute_features import parkinson_compute_features
from hgsprediction.save_data import parkinson_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
# parkinson_cohort = sys.argv[4]
# visit_session = sys.argv[5]
# gender = sys.argv[6]


# if visit_session == "1":
#     session_column = f"1st_{parkinson_cohort}_session"
# elif visit_session == "2":
#     session_column = f"2nd_{parkinson_cohort}_session"
# elif visit_session == "3":
#     session_column = f"3rd_{parkinson_cohort}_session"
# elif visit_session == "4":
#     session_column = f"4th_{parkinson_cohort}_session"

# df = parkinson_load_data.load_validated_hgs_data(population, mri_status, session_column, gender="both_gender")

# df = parkinson_compute_features.compute_features(df, session_column, feature_type)

# df_female = df[df["31-0.0"]==0.0]
# df_male = df[df["31-0.0"]==1.0]

# parkinson_save_data.save_preprocessed_data(df, population, mri_status, session_column, "both_gender")
# parkinson_save_data.save_preprocessed_data(df_female, population, mri_status, session_column, "female")
# parkinson_save_data.save_preprocessed_data(df_male, population, mri_status, session_column, "male")

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
if mri_status == "mri":
    visit_range = range(1, 3)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
# for all session pre- and -post parkinson together (all in one):
for parkinson_cohort in ["pre-parkinson", "post-parkinson", "longitudinal-parkinson"]:
    if parkinson_cohort == "longitudinal-parkinson":
        for visit_session in range(1, 2):
            if visit_session == 1:
                session_column = f"1st_{parkinson_cohort}_session"
            df = parkinson_load_data.load_preprocessed_data(population, mri_status, session_column, parkinson_cohort)
            for parkinson_subgroup in ["pre-parkinson", "post-parkinson"]:
                if visit_session == 1:
                    subgroup_session_column = f"1st_{parkinson_subgroup}_session"
                df = parkinson_compute_features.compute_features(df, subgroup_session_column, feature_type, mri_status)
                print(df)
            parkinson_save_data.save_preprocessed_data(df, population, mri_status, session_column, parkinson_cohort)
            print(parkinson_cohort)
    else:
        for visit_session in visit_range:
            if visit_session == 1:
                session_column = f"1st_{parkinson_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{parkinson_cohort}_session"
            elif visit_session == 3:
                session_column = f"3rd_{parkinson_cohort}_session"
            df = parkinson_load_data.load_preprocessed_data(population, mri_status, session_column, parkinson_cohort)
            df = parkinson_compute_features.compute_features(df, session_column, feature_type, mri_status)
        
            parkinson_save_data.save_preprocessed_data(df, population, mri_status, session_column, parkinson_cohort)
            print(parkinson_cohort)
            print(df)

print("===== Done! =====")
embed(globals(), locals())