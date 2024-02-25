
import pandas as pd
from scipy.stats import pearsonr
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def calculate_pearson_hgs_correlation(df, y_axis, x_axis):
   
    y_axis_columns = [col for col in df.columns if any(item in col for item in y_axis)]
    x_axis_columns = [col for col in df.columns if any(item in col for item in x_axis)]

    df_corr = pd.DataFrame(index=y_axis_columns, columns=x_axis_columns)
    df_pvalue = pd.DataFrame(index=y_axis_columns, columns=x_axis_columns)
    
    for y_item in y_axis_columns:
        for x_item in x_axis_columns:
            corr, pvalue = pearsonr(df.loc[:, y_item], df.loc[:, x_item])
            df_corr.loc[y_item, x_item] = corr
            df_pvalue.loc[y_item, x_item] = pvalue

    return df_corr, df_pvalue