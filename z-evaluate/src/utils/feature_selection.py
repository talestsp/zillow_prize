import pandas as pd
from os import listdir
from src.dao.dao import DAO
from src.utils.path_manager import PathManager

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def select_by_corr_thresh(df, corr_threshold=0.75):

    df_corr = df.corr().sort_values(by="logerror", ascending=False)
    df_corr = abs(df_corr)
    # print(df_corr)
    # print()

    #selecting k best
    use_df_corr = df_corr.head(int(len(df_corr) / 2))
    use_df_corr = use_df_corr[use_df_corr.index.tolist()]

    # print(use_df_corr)
    # print(use_df_corr.shape)

    good_cols = use_df_corr.index.tolist()
    good_cols.remove("logerror")
    picked_cols = []

    for index, row in use_df_corr.loc[good_cols][good_cols].iterrows():
        # print(index)
        use_row = row[row.index != index]
        high_correlateds = use_row[use_row > corr_threshold].index.tolist()
        for high_correlated in high_correlateds:
            if high_correlated in good_cols and not high_correlated in picked_cols:
                good_cols.remove(high_correlated)

        picked_cols.append(index)

    return good_cols


if __name__ == "__main__":
    new_features_list = listdir(PathManager().get_new_features_dir())
    new_features_list = [[new_features.replace(".csv", "")] for new_features in new_features_list]
    print("new_features_list:", new_features_list)

    dao = DAO(df_file_name="train_complete_2016.csv", new_features=["knn-longitude-latitude"])
    df = dao.get_normalized_data(max_na_count_columns=0.05)
    df = df.dropna()

    print(select_by_corr_thresh(df))


#good_cols: ['longitude--latitude', 'bedroomcnt', 'structuretaxvaluedollarcnt', 'yearbuilt']