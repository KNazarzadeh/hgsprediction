
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats import multitest

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlations, df, y_axis, x_axis):

    # y_axis_columns = [col for col in df.columns if any(item in col for item in y_axis)]
    
    correlation_values = pd.DataFrame(columns=["regions", "correlations", "p_values"])

    for i, region in enumerate(x_axis):
        # Compute correlations and p-values for all, female, and male datasets
        corr, p_value = pearsonr(df_brain_correlations.iloc[:, i].values.ravel(), df[y_axis].values.ravel())
        correlation_values.loc[i, "regions"] = region
        correlation_values.loc[i, "correlations"] = corr
        correlation_values.loc[i, "p_values"] = p_value
    
    
    # Perform FDR correction on p-values
    reject, p_corrected, _, _ = multitest.multipletests(correlation_values['p_values'], method='fdr_bh')
    
    # Add corrected p-values and significance indicator columns to dataframes
    correlation_values['pcorrected'] = p_corrected
    correlation_values['significant'] = reject

    
    correlation_values_significant = correlation_values[correlation_values['significant']==True]
    
    
    return correlation_values, correlation_values_significant