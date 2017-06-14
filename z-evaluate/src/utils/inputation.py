from sklearn.preprocessing import Imputer
import gc

def inputation(df, strategy="most_frequent"):
    inputer = Imputer(strategy=strategy, axis=0)
    #TODO

def col_mean_inputer(df):
    df_inputed = df.copy()
    for col in df_inputed.columns.tolist():
        col_mean = df_inputed[col].mean()
        df_inputed[col] = df_inputed[col].fillna(col_mean)

    gc.collect()
    return df_inputed

