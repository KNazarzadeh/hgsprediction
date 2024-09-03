import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.define_features import define_features
from hgsprediction.load_data.disorder import load_disorder_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
disorder_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
gender = sys.argv[7]
first_event = sys.argv[8]

##############################################################################
# Define main features and extra features:
features, extend_features = define_features(feature_type)
##############################################################################
# Define X as main features and y as target:
X = features
y = target

##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
###############################################################################

df = load_disorder_data.load_extracted_data_by_feature_and_target(
        population,
        mri_status,
        session_column,
        feature_type,
        target,
        gender,
        first_event,
    )

###############################################################################
pre_subcohort = f"1st_pre-{population}_"
post_subcohort = f"1st_post-{population}_"

subgroup_columns_pre = [col for col in df.columns if pre_subcohort in col]

subgroup_columns_post = [col for col in df.columns if post_subcohort in col]

df_pre = df[subgroup_columns_pre]
df_post = df[subgroup_columns_post]

# Remove 'pre_subcohort' from all column names
df_pre.columns = df_pre.columns.str.replace(pre_subcohort, '', regex=False)
# Remove 'pre_subcohort' from all column names
df_post.columns = df_post.columns.str.replace(post_subcohort, '', regex=False)

interested_columns = X + [y] + ['handedness'] + ['years']

df_pre = df_pre[interested_columns]
df_post = df_post[interested_columns]

###############################################################################
print("Gender:", gender)
print("\n Number of Pre data N=", len(df_pre))
print("\n Number of Post data N=", len(df_post))


summary_stats_pre = df_pre.describe().apply(lambda x: round(x, 2))
summary_stats_post = df_post.describe().apply(lambda x: round(x, 2))

print("summary_stats_pre:\n", summary_stats_pre)
print("###############################################################################")
print("summary_stats_post:\n", summary_stats_post)

pre_right_handed = len(df_pre[df_pre['handedness']==1.0])
post_right_handed = len(df_post[df_post['handedness']==1.0])

print("###############################################################################")
print("N of Pre Right dominant handed =", pre_right_handed)
print("N of Pre Right dominant handed =", post_right_handed)


print("'%' of Pre Right dominant handed =", "{:.2f}".format(pre_right_handed*100/len(df_pre)))
print("'%' of Post Right dominant handed =", "{:.2f}".format(post_right_handed*100/len(df_post)))

print("===== Done! =====")
embed(globals(), locals())
