
import pandas as pd
import numpy as np
import sys
from hgsprediction.load_data.healthy import load_healthy_data
####### Data Extraction #######
from hgsprediction.save_data.healthy import save_multi_samplesize_training_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###################################################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
confound_status = sys.argv[5]
data_set = sys.argv[6]
gender = sys.argv[7]
###################################################################################################
session="0"
# Read ready training data 
data_extracted = load_healthy_data.load_extracted_data_by_feature_and_target(
    population,
    mri_status,
    feature_type,
    target,
    session,
    gender,
    data_set,
) 
# a list of samplesize based on the samplesize/percentage_step (e.g. 10%)
# Take percentage_step% (e.g. 10%) sample from the entire data
percentage_step = 10
# Initialize a list to store multiple percentage_step% (e.g. 10%) samples
samplesize_list = list()
# Take percentage_step% (e.g. 10%) sample from the entire data
df_percent = data_extracted.sample(frac=1/percentage_step, random_state=47)
# Get the number of samples in the initial percentage_step% (e.g. 10%) sample
n_sample_10_percent = len(df_percent)
# Create a temporary DataFrame without the initial percentage_step% (e.g. 10%) sample
df_tmp = data_extracted[~data_extracted.index.isin(df_percent.index)]

# concat initial sample to the list of samples
samplesize_list.insert(0, df_percent)

# Loop to create 9 additional percentage_step% (e.g. 10%) samples
for i in range(1, 10):
    if i == 9:
        # If it's the last iteration, use the entire remaining data as the percentage_step% sample
        df_percent = df_tmp
        samplesize_list.insert(i, df_percent)
    else:
        # Take 10% sample number from the rest of data
        df_percent = df_tmp.sample(n=n_sample_10_percent, random_state=47)
        # concat new sample to the list of samples
        samplesize_list.insert(i, df_percent)
        # Remove the selected percentage_step% (e.g. 10%) sample from the temporary DataFrame
        df_tmp = df_tmp[~df_tmp.index.isin(df_percent.index)]
      
###################################################################################################
# list_of_interest contains percentage values like [10, 20, 40, 60, 80, 100]
list_of_interest = [10, 20, 40, 60, 80, 100]

for percent_interest in list_of_interest:
    df_sample = pd.DataFrame()
    samplesize = f"{percent_interest}_percent"
    num_samples = int(percent_interest / 10)
    for sample_index in range(num_samples):
        # Concatenate individual samples to create a cumulative sample
        df_sample = pd.concat([df_sample, samplesize_list[sample_index]])
        print('num_samples', num_samples)
        print('sample_index', sample_index)
        print(df_sample)
          
        # Now, df_sample contains cumulative samples for each percentage in list_of_interest
        # Ready to save
        save_multi_samplesize_training_data(
            df_sample,
            population,
            mri_status,
            confound_status,
            gender,
            feature_type,
            target,
            session,
            data_set,
            samplesize,
        )

print("===== Done! =====")
embed(globals(), locals())        

