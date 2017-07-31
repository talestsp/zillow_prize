from sklearn.preprocessing import Imputer
import gc


def col_mean_inputer(df):
    #input value with the mean of the column if numeric.
    #if not it random picks a non na value from the serie

    df_inputed = df.copy()
    for col_name in df_inputed.columns.tolist():
        col = df_inputed[col_name]

        try:
            col.astype(int)
            fill_col_value = col.mean()

        except:
            fill_col_value = col.dropna().sample(1).item()

        df_inputed[col_name] = df_inputed[col_name].fillna(fill_col_value)

    gc.collect()
    return df_inputed

