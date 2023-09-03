import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import save_spearman_correlation_results
from hgsprediction.load_results.healthy import load_hgs_predicted_results
from hgsprediction.load_results.healthy import load_spearman_correlation_results
from hgsprediction.save_plot.save_correlations_plot import healthy_save_correlations_plot
from hgsprediction.plots.plot_correlations import healthy_plot_hgs_correlations

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
y = sys.argv[6]
x = sys.argv[7]

###############################################################################
df = load_hgs_predicted_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
)

df_corr, df_pvalue = load_spearman_correlation_results (population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
)
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = sns.load_dataset("iris")

# Define x and y variables
x = "sepal_length"
y = "sepal_width"

# Create a JointGrid
g = sns.JointGrid(data=data, x=x, y=y)

# Hexbin plot 1
joint1 = sns.jointplot(data=data, x=x, y=y, kind="hex", color="blue", ax=g.ax_joint)

# Hexbin plot 2
joint2 = sns.jointplot(data=data, x=x, y=y, kind="hex", color="red", ax=joint1.ax_joint)

# Close the empty scatter plots created by jointplot
joint1.ax_joint.get_children()[0].remove()
# joint2.ax_joint.get_children()[0].remove()

# Add marginal histograms or KDE plots
# sns.histplot(data=data, x=x, ax=g.ax_marg_x, color="lightgray", kde=True)
# sns.histplot(data=data, y=y, ax=g.ax_marg_y, color="lightgray", kde=True)

# Show the plot
plt.show()

plt.savefig("xc.png")
###############################################################################
healthy_plot_hgs_correlations.plot_hgs_correlations_hexbin_plot(
    df, 
    x,
    y,
    population,
    mri_status,
    model_name,
    feature_type,
    target,)

print("===== Done! =====")
embed(globals(), locals())