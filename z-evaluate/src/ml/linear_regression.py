from sklearn import linear_model


class LinearRegression:

    def __init__(self, normalize=False):
        self.normalize = normalize
        self.linear_regression = linear_model.LinearRegression(normalize=normalize)

        self.MODEL_NAME = "LinearRegression"
        print("model:", self.MODEL_NAME)

    def train(self, df_train, target_name):
        use_cols = df_train.columns.tolist()
        use_cols.remove(target_name)

        self.linear_regression.fit(X=df_train[use_cols].values, y=df_train[target_name].values)

        self.df_train = df_train
        self.target_name = target_name

    def get_model_name(self):
        return self.MODEL_NAME

    def predict(self, df_test):
        return self.linear_regression.predict(df_test)

    def r2(self):
        train_cols = self.df_train.columns.tolist()
        train_cols.remove(self.target_name)

        r2 = self.linear_regression.score(self.df_train[train_cols], self.df_train[self.target_name])
        return r2

    def params(self):
        return self.linear_regression.get_params(deep=True)


class RANSACRegression(LinearRegression):

    def __init__(self):
        self.linear_regression = linear_model.RANSACRegressor()

        self.MODEL_NAME = "RANSACRegression"
        print("model:", self.MODEL_NAME)





