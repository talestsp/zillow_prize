import pandas as pd
from sklearn import preprocessing
import gc

from src.utils.path_manager import PathManager

TRAIN_2016_DATA_FILE_NAME = "train_complete_2016.csv"

class DAO:

    def __init__(self, df_file_name=TRAIN_2016_DATA_FILE_NAME, new_features=[]):
        self.pm = PathManager()

        df_file_path = self.pm.get_data_dir(df_file_name)
        self.data = self.load_data(df_file_path, new_features=new_features)

    def load_data(self, df_file_path, new_features=[]):
        df = pd.read_csv(df_file_path)
        df = df.set_index(df["parcelid"])
        del df["parcelid"]

        for new_feature in new_features:
            path = PathManager().get_new_features_dir() + new_feature + ".csv"
            new_feature_df = pd.read_csv(path)
            new_feature_df = new_feature_df.set_index(new_feature_df["parcelid"])

            df = df.merge(new_feature_df, left_index=True, right_index=True, how="left")

        gc.collect()
        return df

    def get_data(self, cols_type=None, max_na_count_columns=1):
        '''

        cols_type: None or 'numeric' values are accepted.
                None: returns all columns
                'numeric': returns only numeric columns

        max_na_count_columns: Set the NAs threshold for the maximum NAs proportion.
                Example: 1 to return columns that have NAs proportion less or equal than 100%
                Example: 0.25 to return columns that have NAs proportion less or equal than 25%

        '''
        if cols_type is None:
            data = self.data

        elif cols_type == "numeric":
            numeric_cols = self.infer_numeric_cols(self.data)
            data = self.data[numeric_cols]

        use_cols = self.less_na_cols(data, threshold=max_na_count_columns)

        return data[use_cols]

    def get_normalized_data(self, max_na_count_columns=1):
        '''
        Returns normalize data.
        Only numeric data will be returned.
        Rows with any NA values are removed.

        max_na_count_columns: Set the NAs threshold for the maximum NAs proportion.
                Example: 1 to return columns that have NAs proportion less or equal than 100%
                Example: 0.25 to return columns that have NAs proportion less or equal than 25%
        '''
        df = self.get_data(cols_type="numeric", max_na_count_columns=max_na_count_columns)
        df = df.dropna()

        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_norm = pd.DataFrame(x_scaled)

        df_norm.columns = df.columns
        gc.collect()
        return df_norm

    def infer_numeric_cols(self, df):
        numeric_cols = []

        for col in df.columns:
            try:
                df[col].astype("float")
                numeric_cols.append(col)
            except ValueError:
                pass

        return numeric_cols

    def less_na_cols(self, data, threshold=1):
        '''
            Return column names with NAs count less or equal than threshold
        '''

        na_df = pd.Series(data.isnull().sum() / len(data)).sort_values(ascending=False)
        cols = na_df[na_df <= threshold].index.tolist()

        return cols
