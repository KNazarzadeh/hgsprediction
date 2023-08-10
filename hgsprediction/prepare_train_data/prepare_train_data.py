import pandas as pd

def prepare_train_data(data):
    
    data = data.drop("index", axis=1)
    data.rename(columns={'eid': 'SubjectID'}, inplace=True)
    data.set_index("SubjectID", inplace=True)

    # Remove unnecessary columns
    data = data.drop(["Age_bins","hgs_bins","gender_bins","mix_bins","bins_prob_num", "hgs(L+R)-0.0", "dominant_hgs-0.0", "nondominant_hgs-0.0"], axis=1)
    # Save the DataFrame to a CSV file with 'SubjectID' as the index
    # Specify 'SubjectID' as the index label
    data.to_csv('data_with_new_index.csv', index_label='SubjectID')
    
    
