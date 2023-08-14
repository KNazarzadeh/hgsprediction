import sys
import pandas as pd
from hgsprediction.compute_target import StrokeTargetComputing


filename = sys.argv[0]
mri_status = sys.argv[1]
feature_type = sys.argv[2],
target = sys.argv[3]
stroke_cohort = sys.argv[3],
visit_session = sys.argv[4]

# df = load_data(mri_status)    
"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/results_hgsprediction/healthy/nonmri/female/anthropometrics_age"
data_processor = StrokeTargetComputing(df, mri_status, feature_type, stroke_cohort, visit_session)

# Call all functions inside the class
# Calculate target
if target == "hgs_L+R":
    df = data_processor.calculate_sum_hgs(df)
elif target == "hgs_left":
    df = data_processor.calculate_left_hgs(df)
elif target == "hgs_right":
    df = data_processor.calculate_right_hgs(df)
elif target == "hgs_dominant":
    df = data_processor.calculate_dominant_hgs(df)
elif target == "hgs_nondominant":
    df = data_processor.calculate_nondominant_hgs(df)
elif target == "hgs_L-R":
    df = data_processor.calculate_sub_hgs(df)
elif target == "hgs_LI":
    df = data_processor.calculate_laterality_index_hgs(df)

