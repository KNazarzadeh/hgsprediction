import sys
import os
import pandas as pd
from hgsprediction.compute_features import StrokeFeaturesComputing
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

motor = "hgs"
mri_status = "mri"
population = "stroke"
feature_type = "anthropometrics_age"
stroke_cohort = "post"
visit_session = 1
post_list = ["1_post_session", "2_post_session", "3_post_session", "4_post_session"]
folder_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "GIT_repositories",
    "motor_ukb",
    "data_ukb",
    f"data_{motor}",
    population,
    "prepared_data",
    f"{mri_status}_{population}",
)

file_path = os.path.join(
        folder_path,
        f"{post_list[0]}_{mri_status}_{population}.csv")

df = pd.read_csv(file_path, sep=',', index_col=0)
df = df.rename(columns={"1_post_session": "1st_post-stroke_session"})
df.loc[:, "1st_post-stroke_session"] = df.loc[:, "1st_post-stroke_session"].str.replace("session-", "")

data_processor = StrokeFeaturesComputing(df, mri_status, feature_type, stroke_cohort, visit_session)

# Call all functions inside the class
# FEATURE ENGINEERING
data = data_processor.calculate_bmi(df)
data = data_processor.calculate_height(data)
data = data_processor.calculate_waist_to_hip_ratio(data)
data = data_processor.calculate_age(data)
data = data_processor.calculate_days(data)

