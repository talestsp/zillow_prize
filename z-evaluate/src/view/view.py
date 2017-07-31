import pandas as pd
import json
from os import listdir
from src.utils.path_manager import PathManager
pd.set_option('display.float_format', lambda x: '%.7f' % x)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data_eval_file_paths = listdir(PathManager().get_results_data_eval_dir())
if len(data_eval_file_paths) == 0:
    raise Exception("No results found")

evals = []

for file_path in data_eval_file_paths:
    with open(PathManager().get_results_data_eval_dir() + file_path, "r") as file:
        content = json.load(file)
        evals.append(content)

evals_df = pd.DataFrame(evals).sort_values(by="mae").reset_index()
evals_df.to_csv(PathManager().get_results_dir() + "evals_df.csv", index=False)

evals_df = pd.DataFrame(evals).sort_values(by="mae", ascending=True)

use_evals = evals_df[(evals_df["abs"].astype(str) != "True") &
                     (evals_df["inputation"] != "drop")]

print("all_cols:", evals_df.columns.tolist())
print()
use_cols = ["cols_type", "feat_selection", "inputation", "model_name", "norm", "abs", "r2", "mae", "new_features", "id"]

print(use_evals[use_cols])




