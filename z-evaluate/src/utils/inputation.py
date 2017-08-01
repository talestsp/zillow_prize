from sklearn.preprocessing import Imputer
import gc


def col_mean_inputer(df):
    #input value with the mean of the column if numeric.
    #if not it uses the most frequent value
    print("col_mean inputation may take few minutes")


    df_inputed = df.copy()
    for col_name in df_inputed.columns.tolist():
        col = df_inputed[col_name]
        print(col_name)

        try:
            col.astype(float)
            fill_col_value = col.mean()
            df_inputed[col_name] = df_inputed[col_name].fillna(fill_col_value)

        except ValueError as e:
            most_frequent = col.value_counts().index[0]
            df_inputed[col_name] = df_inputed[col_name].fillna(most_frequent)

    gc.collect()

    return df_inputed


def random_pick_one(x, col):
    if str(x) == "nan":
        random_x =  col.dropna().sample(1).item()
        return random_x
    else:
        return x

def col_mean_inputer_fine(df):
    print("\n\n\n")
    print("col_mean fine inputation may take few minutes")

    df_inputed = df.copy()
    for col_name in df_inputed.columns.tolist():
        col = df_inputed[col_name]
        print(col_name)

        try:
            col.astype(float)
            fill_col_value = col.mean()
            df_inputed[col_name] = df_inputed[col_name].fillna(fill_col_value)

        except ValueError as e:
            df_inputed[col_name] = col.apply(random_pick_one, args=(col,))

    return df_inputed