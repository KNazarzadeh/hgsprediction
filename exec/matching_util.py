import sys
import numpy as np
import pandas as pd
from psmpy import PsmPy
from hgsprediction.load_results import healthy
from hgsprediction.load_results import stroke

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
def match(data, target, confound, m, seed=None):
    '''
    Generates a matched sample of size m from data.
    data: pandas dataframe
    target: name of the target variable, must be binary (string)
    confound: list of confounding variables to match on (strings)
    m : number of samples to draw
    seed : random seed
    
    Returns: index of matched sample
    '''
    
    if seed is not None:
        np.random.seed(seed)
    
    # randomly select a target
    targets = np.unique(data[target])
    assert(len(targets) == 2) # Currently only two targets supported.
    atarget = np.random.choice(targets)
    # make sure that we can sample from this target    
    n_samp = round(m/2)
    idx = data[target] == atarget
    idx2 = ~idx
    assert(n_samp < np.sum(idx))    
    idx = np.random.choice(np.where(idx)[0], n_samp, replace=False)    
    # combine idx and idx2
    idx = np.concatenate((idx, np.where(idx2)[0]))
    print(len(idx))
    # prepare data for PsmPy    
    col_for_psm = [target]
    col_for_psm.extend(confound)
    print(col_for_psm)
    df = data[col_for_psm]    
    # retain only the sampled rows
    df = df.iloc[idx,:]
    df['index'] = df.index    

    psm = PsmPy(df, treatment=target, indx='index')
    psm.logistic_ps(balance=True)
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)

    index = psm.df_matched['index'].values
    
    return index


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

df_2 = healthy.load_hgs_predicted_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session="2",
)
stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_longitudinal = stroke.load_hgs_predicted_results("stroke", mri_status, session_column, model_name, feature_type, target, "both_gender")
    
# Set a random seed for reproducibility
np.random.seed(47)
my_data = df_2.copy()
print("===== Done! =====")
embed(globals(), locals())
session_column = f"1st_scan_"

# Define the variables for matching
target_variable = 'Treatment'
confounding_variables = ['Age', 'Income']
sample_size = 4  # You can choose the desired size of your matched sample

# Call the 'match' function to create a matched sample
matched_indices = match(my_data, target_variable, confounding_variables, sample_size)

# Retrieve the matched sample from the original dataset
matched_sample = my_data.iloc[matched_indices]

# Display the matched sample
print(matched_sample)
print("===== Done! =====")
embed(globals(), locals())