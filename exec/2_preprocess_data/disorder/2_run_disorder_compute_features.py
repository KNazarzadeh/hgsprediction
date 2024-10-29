import sys
import pandas as pd
from hgsprediction.load_data.disorder import load_disorder_data
from hgsprediction.compute_features import disorder_compute_features
from hgsprediction.save_data.disorder import save_disorder_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
first_event = sys.argv[4]
###############################################################################
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
# print("===== Done! =====")
# embed(globals(), locals())
# for all session pre- and -post disorder together (all in one):
for disorder_cohort in [f"pre-{population}", f"post-{population}", f"longitudinal-{population}"]:
    if disorder_cohort == f"longitudinal-{population}":
        for visit_session in range(1, 2):
            if visit_session == 1:
                session_column = f"1st_{disorder_cohort}_session"
            df = load_disorder_data.load_validated_hgs_data(population, mri_status, session_column, disorder_cohort, first_event)

            for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
                if visit_session == 1:
                    subgroup_session_column = f"1st_{disorder_subgroup}_session"
                df = disorder_compute_features.compute_features(df, subgroup_session_column, feature_type, mri_status)
                print(df)
            print("===== Done! =====")
            embed(globals(), locals())                
            # save_disorder_data.save_preprocessed_data(df, population, mri_status, session_column, disorder_cohort, first_event)
            print(disorder_cohort)
    else:
        for visit_session in visit_range:
            if first_event == "first_report":
                if visit_session == 1:
                    session_column = f"1st_{disorder_cohort}_session"
                elif visit_session == 2:
                    session_column = f"2nd_{disorder_cohort}_session"
                elif visit_session == 3:
                    session_column = f"3rd_{disorder_cohort}_session"
                    if population == "parkinson":
                        break
            elif first_event == "first_diagnosis":
                if visit_session == 1:
                    session_column = f"1st_{disorder_cohort}_session"
                        
            df = load_disorder_data.load_validated_hgs_data(population, mri_status, session_column, disorder_cohort, first_event)
            df = disorder_compute_features.compute_features(df, session_column, feature_type, mri_status)
        
            # save_disorder_data.save_preprocessed_data(df, population, mri_status, session_column, disorder_cohort, first_event)
            print(disorder_cohort)
            print(df)

print("===== Done! =====")
embed(globals(), locals())