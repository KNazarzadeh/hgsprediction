import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import stroke
import statsmodels.api as sm

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_longitudinal = stroke.load_hgs_predicted_results(population, mri_status, session_column, model_name, feature_type, target, "both_gender")
print("===== Done! =====")
embed(globals(), locals())
# Define your age bins/ranges
bins = [44, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]  # Customize the age ranges as needed
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# Use pd.cut() to create the age range column
df_longitudinal["age_range"] = pd.cut(df_longitudinal["1st_pre-stroke_age"], bins=bins, labels=labels)

###############################################################################
def update_weakness_column(df, gender_value, hgs_type):
    
    if gender_value == 0:
        thresholds = {
            "age_ranges": labels,  # Replace with your actual age ranges
            "thresholds": [24, 23, 22, 21, 20, 20, 18, 18, 18, 18]  # Replace with the corresponding threshold values
        }
        
    elif gender_value == 1:
        thresholds = {
            "age_ranges": labels,  # Replace with your actual age ranges
            "thresholds": [38, 36, 36, 34, 33, 32, 30, 30, 30, 30]  # Replace with the corresponding threshold values
        }
    
        for i, age_range in enumerate(thresholds["age_ranges"]):
            mask1 = (df["gender"] == gender_value) & (df["age_range"] == age_range) & (df[f"1st_pre-stroke_{target}_{hgs_type}"] < thresholds["thresholds"][i])
            mask2 = (df["gender"] == gender_value) & (df["age_range"] == age_range) & (df[f"1st_pre-stroke_{target}_{hgs_type}"] >= thresholds["thresholds"][i])
        
            # mask1 = (df_longitudinal["gender"] == gender_value) & (df_longitudinal[f"1st_pre-stroke_{target}_{hgs_type}"] < 18)
            # mask2 = (df_longitudinal["gender"] == gender_value) & (df_longitudinal[f"1st_pre-stroke_{target}_{hgs_type}"] >= 18)
    
            df_longitudinal.loc[mask1, f"1st_pre-post-stroke_hgs_{hgs_type}_weakness"] = (
                df_longitudinal.loc[mask1, f"1st_post-stroke_{target}_{hgs_type}"].apply(lambda x: "weakness_to_weakness" if x < thresholds["thresholds"][i] else "weakness_to_normal"))

            df_longitudinal.loc[mask2, f"1st_pre-post-stroke_hgs_{hgs_type}_weakness"] = (
                df_longitudinal.loc[mask2, f"1st_post-stroke_{target}_{hgs_type}"].apply(lambda x: "normal_to_weakness" if x < thresholds["thresholds"][i] else"normal_to_normal"))

# Apply for gender == 0
update_weakness_column(df_longitudinal, gender_value=0, hgs_type="actual")
# Apply for gender == 1
update_weakness_column(df_longitudinal, gender_value=1, hgs_type="actual")

# Apply for gender == 0
update_weakness_column(df_longitudinal, gender_value=0, hgs_type="predicted")
# Apply for gender == 1
update_weakness_column(df_longitudinal, gender_value=1, hgs_type="predicted")


print("===== Done! =====")
embed(globals(), locals())

###############################################################################
