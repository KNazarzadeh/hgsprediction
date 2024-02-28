import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles


from hgsprediction.load_results.load_brain_correlates_results import load_brain_overlap_data_results, load_brain_hgs_correlation_results


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
session = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
brain_data_type = sys.argv[9]
schaefer = sys.argv[10]
stats_correlation_type = sys.argv[11]
###############################################################################
for gender in ["female", "male"]:
    for corr_target in ["true_hgs", "predicted_hgs", "delta_hgs"]:
        df = pd.DataFrame()
        for target in ["hgs_L+R", "hgs_left", "hgs_right"]:
            df_corr = load_brain_hgs_correlation_results(
                population,
                mri_status,
                model_name,
                feature_type,
                target,
                f"{gender}",
                session,
                confound_status,
                n_repeats,
                n_folds,
                brain_data_type,
                schaefer,
                f"{corr_target}",
            )

            df_corr = df_corr[df_corr['significant']==True]
            df_corr.loc[:, "target"] = target     
            
            df = pd.concat([df, df_corr], axis=0)
        # print("===== Done! =====")
        # embed(globals(), locals())
        # Extract unique regions from each DataFrame
        regions_df1 = set(df[df['target']=="hgs_L+R"]['regions'])
        regions_df2 = set(df[df['target']=="hgs_left"]['regions'])
        regions_df3 = set(df[df['target']=="hgs_right"]['regions'])

        # Find the common items among the three sets
        matching_items = regions_df1.intersection(regions_df2, regions_df3)

        print("Matching Items:")
        for item in matching_items:
            print(item)
        # Create a Venn diagram
        venn3_circles(subsets=(regions_df1, regions_df2, regions_df3),linestyle='dashed', linewidth=2)
        venn3(subsets=(regions_df1, regions_df2, regions_df3), alpha=0.3, set_labels=('HGS_L+R', 'HGS_Left', 'HGS_Right'), set_colors=("red", "blue", "green"))

        # Show the plot
        plt.show()
        plt.savefig(f"venn_{gender}_{corr_target}.png")
        plt.close()

print("===== Done! =====")
embed(globals(), locals())