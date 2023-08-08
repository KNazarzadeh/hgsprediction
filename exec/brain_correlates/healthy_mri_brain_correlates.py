
import sys
import pandas as pd
import numpy as np

from hgsprediction.load_imaging_data import load_imaging_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
img_type = sys.argv[1]
neuroanatomy = sys.argv[2]

img_df = load_imaging_data(img_type,neuroanatomy)



print("===== Done! =====")
embed(globals(), locals())
