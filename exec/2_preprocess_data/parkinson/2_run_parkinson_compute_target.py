import sys
import os
import pandas as pd
from hgsprediction.load_data import parkinson_load_data
from hgsprediction.save_data import parkinson_save_data
from hgsprediction.compute_target import parkinson_compute_target

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

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
            df = parkinson_load_data.load_validated_hgs_data(population, mri_status, session_column, parkinson_cohort)
            for parkinson_subgroup in ["pre-parkinson", "post-parkinson"]:
                if visit_session == 1:
                    subgroup_session_column = f"1st_{parkinson_subgroup}_session"
                for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:
                    df = parkinson_compute_target.compute_target(df, subgroup_session_column, target)
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
            df = parkinson_load_data.load_validated_hgs_data(population, mri_status, session_column, parkinson_cohort)
            for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:
                df = parkinson_compute_target.compute_target(df, session_column, target)

            parkinson_save_data.save_preprocessed_data(df, population, mri_status, session_column, parkinson_cohort)

print("===== Done! =====")
embed(globals(), locals())