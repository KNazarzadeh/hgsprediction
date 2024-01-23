
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from hgsprediction.load_data import healthy_load_data
####### Data Extraction #######
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.save_data import save_multi_samplesize_training_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
confound_status = sys.argv[5]
model_name = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
gender = sys.argv[9]


# Read ready training data 
df_train = healthy_load_data.load_ready_training_data(population, mri_status, feature_type, target, confound_status, gender)


# a list of samplesize based on the samplesize/percentage_step (e.g. 10%)
# Take percentage_step% (e.g. 10%) sample from the entire data

percentage_step = 10

# Initialize a list to store multiple percentage_step% (e.g. 10%) samples
samplesize_list = list()

# Take percentage_step% (e.g. 10%) sample from the entire data
df_percent = df_train.sample(frac=1/percentage_step, random_state=47)
# Get the number of samples in the initial percentage_step% (e.g. 10%) sample
n_sample_10_percent = len(df_percent)
# Create a temporary DataFrame without the initial percentage_step% (e.g. 10%) sample
df_tmp = df_train[~df_train.index.isin(df_percent.index)]

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
# Assuming samplesize_list contains your individual samples

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
            model_name,
            n_repeats,
            n_folds,
            samplesize,
        )

print("===== Done! =====")
embed(globals(), locals())        

