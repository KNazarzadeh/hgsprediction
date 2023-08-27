
import sys
import pandas as pd
from hgsprediction.load_data import healthy_load_data
from hgsprediction.save_data import healthy_save_data
from hgsprediction.compute_target import healthy_compute_target
from ptpython.repl import embed


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
gender = sys.argv[3]
target = sys.argv[4]
        
df = healthy_load_data.load_preprocessed_train_data(population, mri_status, gender)

df = healthy_compute_target.compute_target(df, mri_status, target)

healthy_save_data.save_preprocessed_train_data(df, population, mri_status, gender)

print("===== Done! =====")
embed(globals(), locals())