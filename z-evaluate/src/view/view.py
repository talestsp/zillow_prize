import pandas as pd
import json
from os import listdir
from src.utils.path_manager import PathManager
pd.set_option('display.float_format', lambda x: '%.6f' % x)


data_eval_file_paths = listdir(PathManager().get_results_data_eval_dir())
if len(data_eval_file_paths) == 0:
    raise Exception("No results found")

evals = []

for file_path in data_eval_file_paths:
    with open(PathManager().get_results_data_eval_dir() + file_path, "r") as file:
        content = json.load(file)
        evals.append(content)

evals_df = pd.DataFrame(evals).sort_values(by="mae").reset_index()

print(evals_df[['model_name', 'mae', 'r2', 'tags', "new_features", "id"]])

