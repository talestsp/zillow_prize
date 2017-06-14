from sklearn.preprocessing import Imputer
import gc


def col_mean_inputer(df):
    #input value with the mean of the column
    df_inputed = df.copy()
    for col in df_inputed.columns.tolist():
        col_mean = df_inputed[col].mean()
        df_inputed[col] = df_inputed[col].fillna(col_mean)

    gc.collect()
    return df_inputed

