


import pandas as pd
from sys import argv
import numpy as np
from hgsprediction.load_data import load_original_data
from hgsprediction.prepare_data.prepare_disease import PrepareDisease
from hgsprediction.save_data import save_prepared_disease_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = argv[0]
motor = argv[1]
population = argv[2]
mri_status = argv[3]

df_original = load_original_data(motor=motor, population=population, mri_status=mri_status)


###############################################################################

prepare_data = PrepareDisease(df_original, population)
df_available_disease_dates = prepare_data.remove_missing_disease_dates(df_original)
df_available_hgs = prepare_data.remove_missing_hgs(df_available_disease_dates)
df_subtype = prepare_data.define_stroke_type(df_available_hgs)
df_followup_days = prepare_data.define_followup_days(df_available_hgs)
pre_post_long_df = prepare_data.define_pre_post_longitudinal(df_followup_days)
pre_disease_df = prepare_data.extract_pre_disease(df_followup_days)
post_disease_df = prepare_data.extract_post_disease(df_followup_days)
longitudinal_df = prepare_data.extract_longitudinal_disease(df_followup_days)

print("===== Done! =====")
embed(globals(), locals())



# initialize data length of lists.
index_list = ['original_data_both_gender',
              'original_data_female',
              'original_data_male',
              'available_disease_dates_both_gender',
              'available_disease_dates_female',
              'available_disease_dates_male',
              'available_hgs_both_gender',
              'available_hgs_female',
              'available_hgs_male',
              'pre_disease_both_gender',
              'pre_disease_female',
              'pre_disease_male',
              'post_disease_both_gender',
              'post_disease_female',
              'post_disease_male',
              'longitudinal_disease_both_gender',
              'longitudinal_disease_female',
              'longitudinal_disease_male',
                ]

summary_data = pd.DataFrame(columns=['length_of_data'], index=index_list)

summary_data.loc['original_data_both_gender']['length_of_data'] = len(data_original)
summary_data.loc['original_data_female']['length_of_data'] = len(data_original[data_original['31-0.0']==0.0])
summary_data.loc['original_data_male']['length_of_data'] = len(data_original[data_original['31-0.0']==1.0])
summary_data.loc['available_disease_dates_both_gender']['length_of_data'] = len(df_available_disease_dates)
summary_data.loc['available_disease_dates_female']['length_of_data'] = len(df_available_disease_dates[df_available_disease_dates['31-0.0']==0.0])
summary_data.loc['available_disease_dates_male']['length_of_data'] = len(df_available_disease_dates[df_available_disease_dates['31-0.0']==1.0])
summary_data.loc['available_hgs_both_gender']['length_of_data'] = len(df_available_hgs)
summary_data.loc['available_hgs_female']['length_of_data'] = len(df_available_hgs[df_available_hgs['31-0.0']==0.0])
summary_data.loc['available_hgs_male']['length_of_data'] = len(df_available_hgs[df_available_hgs['31-0.0']==1.0])
summary_data.loc['pre_disease_both_gender']['length_of_data'] = len(pre_disease_df)
summary_data.loc['pre_disease_female']['length_of_data'] = len(pre_disease_df[pre_disease_df['31-0.0']==0.0])
summary_data.loc['pre_disease_male']['length_of_data'] = len(pre_disease_df[pre_disease_df['31-0.0']==1.0])
summary_data.loc['post_disease_both_gender']['length_of_data'] = len(post_disease_df)
summary_data.loc['post_disease_female']['length_of_data'] = len(post_disease_df[post_disease_df['31-0.0']==0.0])
summary_data.loc['post_disease_male']['length_of_data'] = len(post_disease_df[post_disease_df['31-0.0']==1.0])
summary_data.loc['longitudinal_disease_both_gender']['length_of_data'] = len(longitudinal_df)
summary_data.loc['longitudinal_disease_female']['length_of_data'] = len(longitudinal_df[longitudinal_df['31-0.0']==0.0])
summary_data.loc['longitudinal_disease_male']['length_of_data'] = len(longitudinal_df[longitudinal_df['31-0.0']==1.0])

print("===== Done! =====")
embed(globals(), locals())
save_prepared_disease_data(df_available_disease_dates, "available_disease_dates", motor, population, mri_status)
save_prepared_disease_data(df_available_hgs, "available_hgs", motor, population, mri_status)
save_prepared_disease_data(pre_disease_df, "pre_disease", motor, population, mri_status)
save_prepared_disease_data(post_disease_df, "post_disease", motor, population, mri_status)
save_prepared_disease_data(longitudinal_df, "longitudinal_disease", motor, population, mri_status)
save_prepared_disease_data(summary_data, "summary_data", motor, population, mri_status)


print("===== Done! =====")
embed(globals(), locals())