import sys
import os
import pandas as pd
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


# Create a boxplot
sns.boxplot(x="day", y="total_bill", data=df)

# Add labels and title
plt.xlabel("Day of the week")
plt.ylabel("Total Bill Amount")
plt.title("Boxplot of Total Bill Amount by Day")

# Show the plot
plt.show()




print("===== Done! =====")
embed(globals(), locals())