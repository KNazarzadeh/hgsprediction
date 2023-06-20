


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
pre_post_long_df = prepare_data.define_pre_post_sessions(pre_post_long_df)
pre_disease_df = prepare_data.extract_pre_disease(df_followup_days)
post_disease_df = prepare_data.extract_post_disease(df_followup_days)
longitudinal_df = prepare_data.extract_longitudinal_disease(df_followup_days)
recovery_df = prepare_data.define_recovery_data(pre_post_long_df)
all_post_stroke = prepare_data.define_all_post_subjects(pre_post_long_df)
hgs_df = prepare_data.define_hgs(pre_post_long_df)

df_available_1_post = hgs_df[((~hgs_df[f"1_post_left_hgs"].isna()) & (hgs_df[f"1_post_left_hgs"] !=  0))
                        & ((~hgs_df[f"1_post_right_hgs"].isna()) & (hgs_df[f"1_post_right_hgs"] !=  0))
                       ]
df_available_2_post = hgs_df[((~hgs_df[f"2_post_left_hgs"].isna()) & (hgs_df[f"2_post_left_hgs"] !=  0))
                        & ((~hgs_df[f"2_post_right_hgs"].isna()) & (hgs_df[f"2_post_right_hgs"] !=  0))
                       ]
df_available_3_post = hgs_df[((~hgs_df[f"3_post_left_hgs"].isna()) & (hgs_df[f"3_post_left_hgs"] !=  0))
                        & ((~hgs_df[f"3_post_right_hgs"].isna()) & (hgs_df[f"3_post_right_hgs"] !=  0))
                       ]
df_available_4_post = hgs_df[((~hgs_df[f"4_post_left_hgs"].isna()) & (hgs_df[f"4_post_left_hgs"] !=  0))
                        & ((~hgs_df[f"4_post_right_hgs"].isna()) & (hgs_df[f"4_post_right_hgs"] !=  0))
                       ]

df_available_1_pre = hgs_df[((~hgs_df[f"1_pre_left_hgs"].isna()) & (hgs_df[f"1_pre_left_hgs"] !=  0))
                        & ((~hgs_df[f"1_pre_right_hgs"].isna()) & (hgs_df[f"1_pre_right_hgs"] !=  0))
                       ]
df_available_2_pre = hgs_df[((~hgs_df[f"2_pre_left_hgs"].isna()) & (hgs_df[f"2_pre_left_hgs"] !=  0))
                        & ((~hgs_df[f"2_pre_right_hgs"].isna()) & (hgs_df[f"2_pre_right_hgs"] !=  0))
                       ]
df_available_3_pre = hgs_df[((~hgs_df[f"3_pre_left_hgs"].isna()) & (hgs_df[f"3_pre_left_hgs"] !=  0))
                        & ((~hgs_df[f"3_pre_right_hgs"].isna()) & (hgs_df[f"3_pre_right_hgs"] !=  0))
                       ]
df_available_4_pre = hgs_df[((~hgs_df[f"4_pre_left_hgs"].isna()) & (hgs_df[f"4_pre_left_hgs"] !=  0))
                        & ((~hgs_df[f"4_pre_right_hgs"].isna()) & (hgs_df[f"4_pre_right_hgs"] !=  0))
                       ]

# print("===== Done! =====")
# embed(globals(), locals())
# initialize data length of lists.
index_list = ['Total_original_data_subjects',
              'original_data_female',
              'original_data_male',
              'Total_available_disease_dates_subjects',
              'available_disease_dates_female',
              'available_disease_dates_male',
              'Total_available_hgs_subjects',
              'available_hgs_female',
              'available_hgs_male',
              'Total_pre_disease_subjects',
              'pre_disease_female',
              'pre_disease_male',
              'Total_post_disease_subjects',
              'post_disease_female',
              'post_disease_male',
              'Total_longitudinal_disease_subjects',
              'longitudinal_disease_female',
              'longitudinal_disease_male',
              'Total_recovery_followup_subjects',              
              'recovery_followup_female',
              'recovery_followup_male',
              'Total_all_post_subjects',              
              'all_post_female',
              'all_post_male',
              'Total_1_post_subjects',              
              '1_post_female',
              '1_post_male',       
              'Total_2_post_subjects',              
              '2_post_female',
              '2_post_male',                    
              'Total_3_post_subjects',              
              '3_post_female',
              '3_post_male',                    
              'Total_4_post_subjects',              
              '4_post_female',
              '4_post_male',          
              'Total_1_pre_subjects',              
              '1_pre_female',
              '1_pre_male',       
              'Total_2_pre_subjects',              
              '2_pre_female',
              '2_pre_male',                    
              'Total_3_pre_subjects',              
              '3_pre_female',
              '3_pre_male',                    
              'Total_4_pre_subjects',              
              '4_pre_female',
              '4_pre_male',               
                ]

summary_data = pd.DataFrame(columns=['length_of_data'], index=index_list)

summary_data.loc['Total_original_data_subjects'] ['length_of_data'] = len(df_original)
summary_data.loc['original_data_female']['length_of_data'] = len(df_original[df_original['31-0.0']==0.0])
summary_data.loc['original_data_male']['length_of_data'] = len(df_original[df_original['31-0.0']==1.0])
summary_data.loc['Total_available_disease_dates_subjects']['length_of_data'] = len(df_available_disease_dates)
summary_data.loc['available_disease_dates_female']['length_of_data'] = len(df_available_disease_dates[df_available_disease_dates['31-0.0']==0.0])
summary_data.loc['available_disease_dates_male']['length_of_data'] = len(df_available_disease_dates[df_available_disease_dates['31-0.0']==1.0])
summary_data.loc['Total_available_hgs_subjects']['length_of_data'] = len(df_available_hgs)
summary_data.loc['available_hgs_female']['length_of_data'] = len(df_available_hgs[df_available_hgs['31-0.0']==0.0])
summary_data.loc['available_hgs_male']['length_of_data'] = len(df_available_hgs[df_available_hgs['31-0.0']==1.0])
summary_data.loc['Total_pre_disease_subjects']['length_of_data'] = len(pre_disease_df)
summary_data.loc['pre_disease_female']['length_of_data'] = len(pre_disease_df[pre_disease_df['31-0.0']==0.0])
summary_data.loc['pre_disease_male']['length_of_data'] = len(pre_disease_df[pre_disease_df['31-0.0']==1.0])
summary_data.loc['Total_post_disease_subjects']['length_of_data'] = len(post_disease_df)
summary_data.loc['post_disease_female']['length_of_data'] = len(post_disease_df[post_disease_df['31-0.0']==0.0])
summary_data.loc['post_disease_male']['length_of_data'] = len(post_disease_df[post_disease_df['31-0.0']==1.0])
summary_data.loc['Total_longitudinal_disease_subjects']['length_of_data'] = len(longitudinal_df)
summary_data.loc['longitudinal_disease_female']['length_of_data'] = len(longitudinal_df[longitudinal_df['31-0.0']==0.0])
summary_data.loc['longitudinal_disease_male']['length_of_data'] = len(longitudinal_df[longitudinal_df['31-0.0']==1.0])
summary_data.loc['Total_recovery_followup_subjects']['length_of_data'] = len(recovery_df)
summary_data.loc['recovery_followup_female']['length_of_data'] = len(recovery_df[recovery_df['31-0.0']==0.0])
summary_data.loc['recovery_followup_male']['length_of_data'] = len(recovery_df[recovery_df['31-0.0']==1.0])
summary_data.loc['Total_all_post_subjects']['length_of_data'] = len(all_post_stroke)
summary_data.loc['all_post_female']['length_of_data'] = len(all_post_stroke[all_post_stroke['31-0.0']==0.0])
summary_data.loc['all_post_male']['length_of_data'] = len(all_post_stroke[all_post_stroke['31-0.0']==1.0])
summary_data.loc['Total_1_post_subjects']['length_of_data'] = len(df_available_1_post)
summary_data.loc['1_post_female']['length_of_data'] = len(df_available_1_post[df_available_1_post['31-0.0']==0.0])
summary_data.loc['1_post_male']['length_of_data'] = len(df_available_1_post[df_available_1_post['31-0.0']==1.0])
summary_data.loc['Total_2_post_subjects']['length_of_data'] = len(df_available_2_post)
summary_data.loc['2_post_female']['length_of_data'] = len(df_available_2_post[df_available_2_post['31-0.0']==0.0])
summary_data.loc['2_post_male']['length_of_data'] = len(df_available_2_post[df_available_2_post['31-0.0']==1.0])
summary_data.loc['Total_3_post_subjects']['length_of_data'] = len(df_available_3_post)
summary_data.loc['3_post_female']['length_of_data'] = len(df_available_3_post[df_available_3_post['31-0.0']==0.0])
summary_data.loc['3_post_male']['length_of_data'] = len(df_available_3_post[df_available_3_post['31-0.0']==1.0])
summary_data.loc['Total_4_post_subjects']['length_of_data'] = len(df_available_4_post)
summary_data.loc['4_post_female']['length_of_data'] = len(df_available_4_post[df_available_4_post['31-0.0']==0.0])
summary_data.loc['4_post_male']['length_of_data'] = len(df_available_4_post[df_available_4_post['31-0.0']==1.0])
summary_data.loc['Total_1_pre_subjects']['length_of_data'] = len(df_available_1_pre)
summary_data.loc['1_pre_female']['length_of_data'] = len(df_available_1_pre[df_available_1_pre['31-0.0']==0.0])
summary_data.loc['1_pre_male']['length_of_data'] = len(df_available_1_pre[df_available_1_pre['31-0.0']==1.0])
summary_data.loc['Total_2_pre_subjects']['length_of_data'] = len(df_available_2_pre)
summary_data.loc['2_pre_female']['length_of_data'] = len(df_available_2_pre[df_available_2_pre['31-0.0']==0.0])
summary_data.loc['2_pre_male']['length_of_data'] = len(df_available_2_pre[df_available_2_pre['31-0.0']==1.0])
summary_data.loc['Total_3_pre_subjects']['length_of_data'] = len(df_available_3_pre)
summary_data.loc['3_pre_female']['length_of_data'] = len(df_available_3_pre[df_available_3_pre['31-0.0']==0.0])
summary_data.loc['3_pre_male']['length_of_data'] = len(df_available_3_pre[df_available_3_pre['31-0.0']==1.0])
summary_data.loc['Total_4_pre_subjects']['length_of_data'] = len(df_available_4_pre)
summary_data.loc['4_pre_female']['length_of_data'] = len(df_available_4_pre[df_available_4_pre['31-0.0']==0.0])
summary_data.loc['4_pre_male']['length_of_data'] = len(df_available_4_pre[df_available_4_pre['31-0.0']==1.0])

save_prepared_disease_data(df_available_disease_dates, "available_disease_dates", motor, population, mri_status)
save_prepared_disease_data(df_available_hgs, "available_hgs", motor, population, mri_status)
save_prepared_disease_data(pre_disease_df, "pre_disease", motor, population, mri_status)
save_prepared_disease_data(post_disease_df, "post_disease", motor, population, mri_status)
save_prepared_disease_data(longitudinal_df, "longitudinal_disease", motor, population, mri_status)
save_prepared_disease_data(recovery_df, "recovery_disease", motor, population, mri_status)
save_prepared_disease_data(df_available_1_post, "1_post_session", motor, population, mri_status)
save_prepared_disease_data(df_available_2_post, "2_post_session", motor, population, mri_status)
save_prepared_disease_data(df_available_3_post, "3_post_session", motor, population, mri_status)
save_prepared_disease_data(df_available_4_post, "4_post_session", motor, population, mri_status)
save_prepared_disease_data(df_available_1_pre, "1_pre_session", motor, population, mri_status)
save_prepared_disease_data(df_available_2_pre, "2_pre_session", motor, population, mri_status)
save_prepared_disease_data(df_available_3_pre, "3_pre_session", motor, population, mri_status)
save_prepared_disease_data(df_available_4_pre, "4_pre_session", motor, population, mri_status)


save_prepared_disease_data(summary_data, "summary_data", motor, population, mri_status)


print("===== Done! =====")
embed(globals(), locals())