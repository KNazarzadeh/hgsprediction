
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def calculate_brain_hgs(df, y_axis, x_axis, stats_correlation_type):

    # y_axis_columns = [col for col in df.columns if any(item in col for item in y_axis)]
    
    correlation_values = pd.DataFrame(columns=["regions", "correlations", "p_values"])

    for i, region in enumerate(x_axis):
        # Compute correlations and p-values for all, female, and male datasets
        if stats_correlation_type == "pearson":
            corr, p_value = pearsonr(df.loc[:, region], df.loc[:, y_axis])
        elif stats_correlation_type == "spearman":
            corr, p_value = spearmanr(df.loc[:, region], df.loc[:, y_axis])
        correlation_values.loc[i, "regions"] = region
        correlation_values.loc[i, "correlations"] = corr
        correlation_values.loc[i, "p_values"] = p_value
    
    
    # Perform FDR correction on p-values
    reject, p_corrected, _, _ = multipletests(correlation_values.loc[:, 'p_values'], method='bonferroni')
    
    # Add corrected p-values and significance indicator columns to dataframes
    correlation_values.loc[:, 'pcorrected'] = p_corrected
    correlation_values.loc[:, 'significant'] = reject

    
    correlation_values_significant = correlation_values[correlation_values.loc[:, 'significant']==True]
    n_regions_survived = len(correlation_values_significant)
    
    return correlation_values, correlation_values_significant, n_regions_survived