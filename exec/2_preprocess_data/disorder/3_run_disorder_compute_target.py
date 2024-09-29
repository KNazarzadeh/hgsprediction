import sys
import os
import pandas as pd
from hgsprediction.load_data.disorder import load_disorder_data
from hgsprediction.save_data.disorder import save_disorder_data
from hgsprediction.compute_target import disorder_compute_target

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
first_event =sys.argv[4]


if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
# for all session pre- and -post disorder together (all in one):
for disorder_cohort in [f"pre-{population}", f"post-{population}", f"longitudinal-{population}"]:
    if disorder_cohort == f"longitudinal-{population}":
        for visit_session in range(1, 2):
            if visit_session == 1:
                session_column = f"1st_{disorder_cohort}_session"
            df = load_disorder_data.load_preprocessed_data(population, mri_status, session_column, disorder_cohort, first_event)

            for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
                if visit_session == 1:
                    subgroup_session_column = f"1st_{disorder_subgroup}_session"
                for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:
                    df = disorder_compute_target.compute_target(df, subgroup_session_column, target)
            print(df)
            save_disorder_data.save_preprocessed_data(df, population, mri_status, session_column, disorder_cohort, first_event)
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
                    
            df = load_disorder_data.load_preprocessed_data(population, mri_status, session_column, disorder_cohort, first_event)
            for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:
                df = disorder_compute_target.compute_target(df, session_column, target)

            save_disorder_data.save_preprocessed_data(df, population, mri_status, session_column, disorder_cohort, first_event)

print("===== Done! =====")
embed(globals(), locals())
