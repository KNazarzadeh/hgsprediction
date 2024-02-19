
import pandas as pd
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def calculate_t_valuesGMV_HGS(df_brain_correlations, df, y_axis, x_axis, significance_level):

    t_values = pd.DataFrame(columns=["regions", "t_values"])
    
    for i, region in enumerate(x_axis):
        # Perform simple linear regression: HGS ~ Feature_i
        slope, intercept, r_value, p_value, std_err = linregress(df_brain_correlations.iloc[:, i].values.ravel(), df[y_axis].values.ravel())
    
        # Calculate the t-value
        t_values.loc[i, "regions"] = region
        t_values.loc[i, "t_values"] = slope / std_err
        t_values.loc[i, "p_value"] = p_value
        
        # Perform FDR correction
        p_values = [p_value]
        rejected, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=significance_level)

        t_values.loc[i, "p_value_corrected"] = p_values_corrected[0]
        t_values.loc[i, "significant"] = rejected[0]

    
    t_values_significant = t_values[t_values['significant']==True]
    
    
    return t_values, t_values_significant