import os
import sys
import pandas as pd
from hgsprediction.load_data.healthy import load_healthy_data
from hgsprediction.save_data.healthy import save_healthy_data
from hgsprediction.compute_target import healthy_compute_target
from ptpython.repl import embed


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
session = sys.argv[3]
data_set = sys.argv[4]
gender = sys.argv[5]
# target = sys.argv[6]

df = load_healthy_data.load_preprocessed_data(population, mri_status, session, gender, data_set)

for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:  
    df = healthy_compute_target.compute_target(df, mri_status, session, target)

save_healthy_data.save_preprocessed_data(df, population, mri_status, session, gender, data_set)

print(df)
print("===== Done! =====")
embed(globals(), locals())