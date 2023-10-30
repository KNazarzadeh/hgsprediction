import sys
import os
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.save_data import stroke_save_data
from hgsprediction.compute_target import stroke_compute_target

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
# for all session pre- and -post stroke together (all in one):
for stroke_cohort in ["pre-stroke", "post-stroke", "longitudinal-stroke"]:
    if stroke_cohort == "longitudinal-stroke":
        for visit_session in range(1, 2):
            if visit_session == 1:
                session_column = f"1st_{stroke_cohort}_session"
            df = stroke_load_data.load_validated_hgs_data(population, mri_status, session_column, stroke_cohort)
            for stroke_subgroup in ["pre-stroke", "post-stroke"]:
                if visit_session == 1:
                    subgroup_session_column = f"1st_{stroke_subgroup}_session"
                for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:
                    df = stroke_compute_target.compute_target(df, subgroup_session_column, target)
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
            df = stroke_load_data.load_validated_hgs_data(population, mri_status, session_column, stroke_cohort)
            for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:
                df = stroke_compute_target.compute_target(df, session_column, target)

            stroke_save_data.save_preprocessed_data(df, population, mri_status, session_column, stroke_cohort)

print("===== Done! =====")
embed(globals(), locals())