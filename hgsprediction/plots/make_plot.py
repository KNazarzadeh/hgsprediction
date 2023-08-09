
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_regplot(target,
                   x_male,
                   y_male,
                   x_female,
                   y_female,
                   corr_male,
                   corr_female,
                   title,
                   y_label,
):

    fig = plt.figure(figsize=(15,15))
    sns.set_context("poster")
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', labelsize=30)
    sns.regplot(x=x_male, y=y_male, line_kws={"color": "grey"}, scatter_kws={"color": "blue"})
    sns.regplot(x=x_female, y=y_female, scatter_kws={"color": "red"}, line_kws={"color": "grey"})

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    # Add two text annotations to the corner
    text_male = 'r= ' + str(format(corr_male, '.3f'))
    text_female = 'r= ' + str(format(corr_female, '.3f'))

    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    
    plt.text(xmax - 0.15 * xmax, ymax - 0.005 * ymax, text_male, verticalalignment='top',
            horizontalalignment='right', fontsize=24, fontweight="bold", color="blue")
    plt.text(xmax - 0.15 * xmax, ymax - 0.065 * ymax,text_female, verticalalignment='top',
            horizontalalignment='right', fontsize=24, fontweight="bold", color="red")

    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')

    plt.xlabel("Post-stroke years", fontsize=30, fontweight="bold")
    plt.ylabel(y_label, fontsize=30, fontweight="bold")
    plt.title(f"{title} HGS vs Post-stroke years-{target}", fontsize=25, fontweight="bold", y=1.05)
    
    plt.tight_layout()  # Adjust layout to prevent clipping of legend

    plt.show()
    plt.savefig(f"{target}-correlate_{title}_postyears.png")
    plt.close()