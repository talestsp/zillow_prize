from src.dao.dao import DAO
from src.utils.feature_selection import select_by_corr_thresh
from pandas import merge
import gc

TARGET = "logerror"

def process_data(dao, dataset, cols_type, norm, inputation, new_features, feat_selection, max_na_count_columns=0.05):
    if dataset == "test":
        max_na_count_columns = 1.0

    #dao = DAO(new_features=new_features)

    if norm and cols_type == "numeric":
        df = dao.get_normalized_data(dataset=dataset, inputation=inputation, max_na_count_columns=max_na_count_columns)

    elif norm and cols_type == "all":
        df_norm = dao.get_normalized_data(dataset=dataset, inputation=inputation, max_na_count_columns=max_na_count_columns)

        if "parcelid" in df_norm.columns.tolist():
            del df_norm["parcelid"]

        df = dao.get_data(dataset=dataset, inputation=inputation, cols_type=cols_type, max_na_count_columns=max_na_count_columns)

        if "parcelid" in df.columns.tolist():
            del df["parcelid"]

        for numeric_col in df_norm.columns.tolist():
            if numeric_col != TARGET:
                del df[numeric_col]

        if dataset == "train":
            on_cols = ['parcelid', TARGET]
        else:
            on_cols = ['parcelid']

        df = merge(df.reset_index(), df_norm.reset_index(), on=on_cols, how="left").set_index('parcelid')

        df_norm = None
        gc.collect()

    else:
        df = dao.get_data(cols_type=cols_type, dataset=dataset, inputation=inputation, max_na_count_columns=max_na_count_columns)

    if dataset == "train" and not feat_selection is None:
        columns = feat_selection(df) + [TARGET]
        df = df[columns]

    return df