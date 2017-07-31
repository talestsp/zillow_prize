import pandas as pd
from sklearn import preprocessing
#from src.utils.inputation import col_mean_inputer
import gc

#from path_manager import PathManager

TRAIN_2016_DATA_FILE_NAME = "train_complete_2016.csv"
TEST_2016_DATA_FILE_NAME = "properties_2016.csv"

class DAO:

    def __init__(self, train_file_name=TRAIN_2016_DATA_FILE_NAME, test_file_name=TEST_2016_DATA_FILE_NAME, new_features=[]):
        #self.pm = PathManager()

        train_df_file_path = "/home/tales/dev/projects/zillow-zestimate/z-evaluate/data/" + train_file_name
        self.data_train = self.load_data(train_df_file_path, new_features=new_features)

        test_df_file_path = "/home/tales/dev/projects/zillow-zestimate/z-evaluate/data/" + test_file_name
        self.data_test = self.load_data(test_df_file_path, new_features=new_features)

    def load_data(self, df_file_path, new_features=[]):
        df = pd.read_csv(df_file_path, low_memory=False)
        df = df.set_index(df["parcelid"])
        del df["parcelid"]

        for new_feature in new_features:
            print(new_feature)
            path = "/home/tales/dev/projects/zillow-zestimate/z-evaluate/data/new_features/" + new_feature + ".csv"
            new_feature_df = pd.read_csv(path, low_memory=False)
            new_feature_df = new_feature_df.set_index(new_feature_df["parcelid"])

            df = df.merge(new_feature_df, left_index=True, right_index=True, how="left")

        gc.collect()
        return df

    def get_data(self, cols_type=None, dataset="train", max_na_count_columns=1):
        '''

        cols_type: None or 'numeric' values are accepted.
                None: returns all columns
                'numeric': returns only numeric columns

        max_na_count_columns: Set the NAs threshold for the maximum NAs proportion.
                Example: 1 to return columns that have NAs proportion less or equal than 100%
                Example: 0.25 to return columns that have NAs proportion less or equal than 25%

        '''

        if dataset == "train":
            use_data = self.data_train
        elif dataset == "test":
            use_data = self.data_test

        if cols_type == "numeric":
            numeric_cols = self.infer_numeric_cols(use_data)
            use_data = use_data[numeric_cols]

        use_cols = self.less_na_cols(use_data, threshold=max_na_count_columns)
        gc.collect()
        return use_data[use_cols]

    def get_normalized_data(self, dataset="train", inputation="drop", max_na_count_columns=1):
        '''
        Returns normalize data.
        Only numeric data will be returned.

        IMPORTANT: Defatul value for inputation means that remaining ROWS with any NA values are removed.

        max_na_count_columns: Set the NAs threshold for the maximum NAs proportion.
                Example: 1 to return COLUMNS that have NAs proportion less or equal than 100%
                Example: 0.25 to return COLUMNS that have NAs proportion less or equal than 25%
        '''
        df = self.get_data(cols_type="numeric", dataset=dataset, max_na_count_columns=max_na_count_columns)

        if inputation == "drop":
            df = df.dropna()
        elif inputation == "fill_0":
            df = df.fillna(0)
        elif inputation == "column_mean":
            df = col_mean_inputer(df)

        if dataset == "train":
            target = df["logerror"]
            del df["logerror"]

        parcelid_index = df.index

        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_norm = pd.DataFrame(x_scaled)

        df_norm.columns = df.columns
        gc.collect()
        df_norm = df_norm.set_index(parcelid_index)

        df_norm["logerror"] = target.tolist()
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
