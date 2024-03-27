import pandas as pd

def binning_data(
  dataframe,
):
    # Binning data on the specific columns 
    # create 5 bins on Age
    age_label = ['age1', 'age2', 'age3', 'age4', 'age5']
    dataframe['Age_bins'] = pd.qcut(dataframe['21003-0.0'], q=5, labels=age_label)

    # create 5 bins on hgs
    hgs_label = ['hgs(L+R)1', 'hgs(L+R)2', 'hgs(L+R)3', 'hgs(L+R)4', 'hgs(L+R)5']
    dataframe['hgs_bins'] = pd.qcut(dataframe['hgs(L+R)-0.0'], q=5, labels=hgs_label)    

    # create 2 bins on hgs
    gender_label = ['female', 'male']
    dataframe['gender_bins'] = dataframe['31-0.0']
    dataframe['gender_bins'].replace(0, 'female',inplace=True)
    dataframe['gender_bins'].replace(1, 'male',inplace=True)

    # Add new column for combination of bins:
    dataframe["mix_bins"]= dataframe["Age_bins"].astype(str)+"," \
        + dataframe['hgs_bins'].astype(str)+","+dataframe['gender_bins'].astype(str)

    # mix bins
    for i in range(2):
        g = gender_label[i]
        if g == 'female':
            bin_sex = "f"     
        elif g == 'male':
            bin_sex = "m"
        str_g = dataframe[dataframe['gender_bins'] == g]
        l_g = str_g.index.tolist()
        for a in age_label:
            str_a = str_g[str_g['mix_bins'].str.contains(a)]
            l_a = str_a.index.tolist()
            for idx_val in l_a:
                bin_age = str_a.loc[idx_val,'Age_bins']
                bin_hgs = str_a.loc[idx_val,'hgs_bins']
                dataframe.loc[idx_val, 'bins_prob_num'] = bin_sex+bin_age[-1]+bin_hgs[-1]
        
    ########################################
    # elif target == "dominant_hgs":
    #     age_label = ['age1', 'age2', 'age3', 'age4', 'age5']
    #     dataframe['Age_bins'] = pd.qcut(dataframe['21003-0.0'], q=5, labels=age_label)

    #     # create 5 bins on hgs
    #     hgs_label = ['dominant_hgs1', 'dominant_hgs2', 'dominant_hgs3', 'dominant_hgs4', 'dominant_hgs5']
    #     dataframe['hgs_bins'] = pd.qcut(dataframe['dominant_hgs-0.0'], q=5, labels=hgs_label)    

    #     # create 2 bins on hgs
    #     gender_label = ['female', 'male']
    #     dataframe['gender_bins'] = dataframe['31-0.0']
    #     dataframe['gender_bins'].replace(0, 'female',inplace=True)
    #     dataframe['gender_bins'].replace(1, 'male',inplace=True)

    #     # Add new column for combination of bins:
    #     dataframe["mix_bins"]= dataframe["Age_bins"].astype(str)+"," \
    #         + dataframe['hgs_bins'].astype(str)+","+dataframe['gender_bins'].astype(str)

    #     # mix bins
    #     for i in range(2):
    #         g = gender_label[i]
    #         if g == 'female':
    #             bin_sex = "f"     
    #         elif g == 'male':
    #             bin_sex = "m"
    #         str_g = dataframe[dataframe['gender_bins'] == g]
    #         l_g = str_g.index.tolist()
    #         for a in age_label:
    #             str_a = str_g[str_g['mix_bins'].str.contains(a)]
    #             l_a = str_a.index.tolist()
    #             for idx_val in l_a:
    #                 bin_age = str_a.loc[idx_val,'Age_bins']
    #                 bin_hgs = str_a.loc[idx_val,'hgs_bins']
    #                 dataframe.loc[idx_val, 'bins_prob_num'] = bin_sex+bin_age[-1]+bin_hgs[-1]
                    
    # ########################################
    # elif target == "nondominant_hgs":
    #     age_label = ['age1', 'age2', 'age3', 'age4', 'age5']
    #     dataframe['Age_bins'] = pd.qcut(dataframe['21003-0.0'], q=5, labels=age_label)

    #     # create 5 bins on hgs
    #     hgs_label = ['nondominant_hgs1', 'nondominant_hgs2', 'nondominant_hgs3', 'nondominant_hgs4', 'nondominant_hgs5']
    #     dataframe['hgs_bins'] = pd.qcut(dataframe['nondominant_hgs-0.0'], q=5, labels=hgs_label)    

    #     # create 2 bins on hgs
    #     gender_label = ['female', 'male']
    #     dataframe['gender_bins'] = dataframe['31-0.0']
    #     dataframe['gender_bins'].replace(0, 'female',inplace=True)
    #     dataframe['gender_bins'].replace(1, 'male',inplace=True)

    #     # Add new column for combination of bins:
    #     dataframe["mix_bins"]= dataframe["Age_bins"].astype(str)+"," \
    #         + dataframe['hgs_bins'].astype(str)+","+dataframe['gender_bins'].astype(str)

    #     # mix bins
    #     for i in range(2):
    #         g = gender_label[i]
    #         if g == 'female':
    #             bin_sex = "f"     
    #         elif g == 'male':
    #             bin_sex = "m"
    #         str_g = dataframe[dataframe['gender_bins'] == g]
    #         l_g = str_g.index.tolist()
    #         for a in age_label:
    #             str_a = str_g[str_g['mix_bins'].str.contains(a)]
    #             l_a = str_a.index.tolist()
    #             for idx_val in l_a:
    #                 bin_age = str_a.loc[idx_val,'Age_bins']
    #                 bin_hgs = str_a.loc[idx_val,'hgs_bins']
    #                 dataframe.loc[idx_val, 'bins_prob_num'] = bin_sex+bin_age[-1]+bin_hgs[-1]
                  
                  
    return dataframe