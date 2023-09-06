import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from hgsprediction.load_data import stroke_load_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
visit_session = sys.argv[3]

stroke_cohort = "pre-post-stroke"
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"

df = stroke_load_data.load_preprocessed_longitudinal_data(population, mri_status, session_column, "both_gender")


# Assuming you have a DataFrame 'df' with the specified columns
# Extract the relevant columns
df_extract_columns_to_plot = df[["31-0.0", "1st_pre-stroke_hgs_left", "1st_post-stroke_hgs_left", "1st_pre-stroke_hgs_right", "1st_post-stroke_hgs_right", "1st_pre-stroke_hgs_L+R", "1st_post-stroke_hgs_L+R"]]

# Create a list of column groups
column_groups = [
    ("HGS Left", ["1st_pre-stroke_hgs_left", "1st_post-stroke_hgs_left"]),
    ("HGS Right", ["1st_pre-stroke_hgs_right", "1st_post-stroke_hgs_right"]),
    ("HGS L+R", ["1st_pre-stroke_hgs_L+R", "1st_post-stroke_hgs_L+R"])
]

# Initialize an empty list to store the melted DataFrames
melted_dfs = []

# Iterate through column groups and create melted DataFrames
for group_name, group_columns in column_groups:
    # Melt the DataFrame for the current group
    melted_group = pd.melt(df_extract_columns_to_plot, id_vars=["31-0.0"], value_vars=group_columns, var_name="variable")
    
    # Create 'hgs_category' based on 'group_name'
    melted_group['hgs_category'] = group_name
    
    # Create 'stroke_cohort' based on 'variable'
    melted_group['stroke_cohort'] = melted_group['variable'].apply(lambda x: 'pre' if 'pre' in x else ('post' if 'post' in x else None))
    
    # Add a new column based on the combination of 'hgs_category' and 'stroke_cohort'
    melted_group['combine_hgs_stroke_cohort_category'] = group_name + '-' + melted_group['stroke_cohort']
    
    # Create 'gender' based on 'gender' column
    melted_group['gender'] = melted_group['31-0.0'].map({0: 'female', 1: 'male'})
    
    # Drop the original 'variable' column
    melted_group.drop(columns=['31-0.0'], inplace=True)
    
    # Append the melted DataFrame to the list
    melted_dfs.append(melted_group)

print(melted_dfs)

print("===== Done! =====")
embed(globals(), locals())

# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer

# Create the boxplot with the custom palette
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
sns.set(style="whitegrid")
# sns.set_context("poster", font_scale=1.25)
ax = sns.boxplot(x="hgs_category", y="value", hue="stroke_cohort", data=final_df, palette=custom_palette)

# Add labels and title
plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
plt.ylabel("HGS values", fontsize=20, fontweight="bold")
plt.title("pre-stroke and post-stroke HGS values", fontsize=20, fontweight="bold")

ymin , ymax = ax.get_ylim()
ystep = 10
#set y-axis ticks (step size=5)
plt.yticks(np.arange(0, 120, 5))
# Show the plot
plt.legend(title="stroke cohort", loc="upper left")  # Add legend
plt.tight_layout()
plt.show()
plt.savefig("hh.png")


print("===== Done! =====")
embed(globals(), locals())

# Assuming you have a DataFrame 'df' with the specified columns
# Extract the relevant columns
df1 = df[["31-0.0", "1st_pre-stroke_hgs_left", "1st_post-stroke_hgs_left", "1st_pre-stroke_hgs_right", "1st_post-stroke_hgs_right", "1st_pre-stroke_hgs_L+R", "1st_post-stroke_hgs_L+R"]]

# Melt the DataFrame with the specified categories
melted_df = pd.melt(df1, id_vars=["31-0.0"], var_name="variable")

# Create 'hgs_category' based on 'variable'
melted_df['hgs_category'] = melted_df['variable'].apply(lambda x: 'left' if 'left' in x else ('right' if 'right' in x else 'L+R'))

# Create 'stroke_cohort' based on 'variable'
melted_df['stroke_cohort'] = melted_df['variable'].apply(lambda x: 'pre' if 'pre' in x else ('post' if 'post' in x else None))

# Assuming you have a DataFrame 'melted_df' with 'hgs_category' and 'stroke_cohort' columns
# Add a new column based on the combination of 'hgs_category' and 'stroke_cohort'
melted_df['hgs_stroke_cohort_category'] = melted_df['hgs_category'] + '-' + melted_df['stroke_cohort']

# Create 'gender' based on 'gender' column
melted_df['gender'] = melted_df['31-0.0'].map({0: 'female', 1: 'male'})

# Drop the original 'variable' column
melted_df.drop(columns=['31-0.0'], inplace=True)

# Print the melted DataFrame
print(melted_df)

# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#29AB87', '#BC544B'])  # You can use any hex color codes you prefer
# Create the boxplot for 'hgs_category' and 'gender'
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
sns.boxplot(x="hgs_stroke_cohort_category", y="value", data=melted_df, hue="gender", palette=custom_palette)

plt.yticks(np.arange(0, 120, 5))
# # Extract unique values of 'hgs_category' and 'stroke_cohort'
unique_hgs_categories = melted_df["hgs_category"].unique()
unique_stroke_cohorts = melted_df["stroke_cohort"].unique()

# Create xtick labels by repeating the 'stroke_cohort' array for each 'hgs_category'
xticks_labels = [f"{cohort}" for hgs_category in unique_hgs_categories for cohort in unique_stroke_cohorts]
# Set xtick positions and labels
plt.xticks(range(len(xticks_labels)), xticks_labels)


# Add labels and title
plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
plt.ylabel("HGs values", fontsize=20, fontweight="bold")
plt.title("HGS values for genders", fontsize=20, fontweight="bold")
plt.legend(title="Gender", loc="upper left")  # Add legend

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig("hhfm.png")

