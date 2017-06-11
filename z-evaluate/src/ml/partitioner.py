def simple_partition(df, train_proportion):
    '''
        Returns train and test pandas.DataFrames
    '''
    size = int(train_proportion * len(df))

    train_indexes = df.sample(size).index
    train_df = df.loc[train_indexes]
    test_df = df[~df.index.isin(train_indexes)]

    return train_df, test_df



if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../../../data/train_2016.csv")

    train, test = simple_partition(df, 0.7)

    print(len(df))
    print(len(train), len(train) / len(df))
    print(len(test), len(test) / len(df))
