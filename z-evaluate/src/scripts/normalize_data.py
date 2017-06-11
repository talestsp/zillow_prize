import pandas as pd
from sklearn import preprocessing

from src.dao.dao import DAO

dao = DAO()
numeric_df = dao.get_data(cols_type="numeric", max_na_count_columns=0.05)

#remove na rows
numeric_df = numeric_df .dropna()

print(numeric_df.head()[["regionidcity", "calculatedbathnbr", "fullbathcnt"]])

x = numeric_df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

df_norm = pd.DataFrame(df)
df_norm.columns = numeric_df.columns

print(df_norm.head()[["regionidcity", "calculatedbathnbr", "fullbathcnt"]])

