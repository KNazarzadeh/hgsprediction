import sys
import pandas as pd
from hgsprediction.compute_target import StrokeTargetComputing
from hgsprediction.load_data import stroke_load_data
from hgsprediction.save_data import stroke_save_data
from ptpython.repl import embed


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]

if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"
            
            
df = stroke_load_data.load_computed_features_data(population, mri_status, session_column)

data_processor = StrokeTargetComputing(df, mri_status, stroke_cohort, visit_session)

# Call all functions inside the class
# Calculate target
# if target == "hgs_L+R":
df = data_processor.calculate_sum_hgs(df)
# elif target == "hgs_left":
df = data_processor.calculate_left_hgs(df)
# elif target == "hgs_right":
df = data_processor.calculate_right_hgs(df)
# elif target == "hgs_dominant":
df = data_processor.calculate_dominant_nondominant_hgs(df)
# elif target == "hgs_nondominant":
# elif target == "hgs_L-R":
df = data_processor.calculate_sub_hgs(df)
# elif target == "hgs_LI":
df = data_processor.calculate_laterality_index_hgs(df)

stroke_save_data.save_computed_targets_data(df,population, mri_status, session_column)

print("===== Done! =====")
embed(globals(), locals())

