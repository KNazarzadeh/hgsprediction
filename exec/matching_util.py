import numpy as np
import pandas as pd
from psmpy import PsmPy

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