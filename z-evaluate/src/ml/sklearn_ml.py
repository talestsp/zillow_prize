import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing

class SKLearnMlBase:

    def __init__(self):
        print("model built:", type(self.model).__name__)

    def train(self, df_train, target_name):
        use_cols = df_train.columns.tolist() #columns from train
        use_cols.remove(target_name) #remove target from train dataset

        self.model.fit(X=df_train[use_cols].values, y=df_train[target_name].values) #training model

        self.df_train = df_train
        self.target_name = target_name

    def get_model_name(self):
        return self.model_name

    def predict(self, df_test):
        #perform the prediction
        return self.model.predict(df_test)

    def r2(self):
        #returns the model's r2 (r-squared)
        train_cols = self.df_train.columns.tolist()
        train_cols.remove(self.target_name)

        r2 = self.model.score(self.df_train[train_cols], self.df_train[self.target_name])
        return r2

    def params(self):
        #returns the model's parameters, these values are differents, it depends on the model used
        return self.model.get_params(deep=True)

    def columns_relevance(self):
        #each column (attribute) has a different relevance depending on the model used
        cols = self.df_train.columns.tolist()
        cols.remove(self.target_name)

        relevance = pd.DataFrame({"columns": cols, "relevance": self.model.coef_})
        relevance["abs_relevance"] = abs(relevance["relevance"])

        relevance = relevance.sort_values(by="abs_relevance", ascending=False)

        norm_relevance = relevance["abs_relevance"].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(norm_relevance)
        norm_relevance = x_scaled
        relevance["relevance"] = norm_relevance

        del relevance["abs_relevance"]

        return relevance


class SKLearnRANSACRegression(SKLearnMlBase):

    def __init__(self):
        self.model = linear_model.RANSACRegressor(linear_model.LinearRegression())
        self.model_name = "SKLearnRANSACRegression"

        SKLearnMlBase.__init__(self)

    #overwriting from base class
    def columns_relevance(self):
        #each column (attribute) has a different relevance depending on the model used
        cols = self.df_train.columns.tolist()
        cols.remove(self.target_name)

        relevance = pd.DataFrame({"columns": cols, "relevance": self.model.estimator_.coef_})
        relevance["abs_relevance"] = abs(relevance["relevance"])

        relevance = relevance.sort_values(by="abs_relevance", ascending=False)

        norm_relevance = relevance["abs_relevance"].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(norm_relevance)
        norm_relevance = x_scaled
        relevance["relevance"] = norm_relevance

        del relevance["abs_relevance"]

        return relevance

    def params(self):
        #returns the model's parameters, these values are differents, it depends on the model used

        params = self.model.get_params(deep=True)
        del params["base_estimator"]

        return params


class SKLearnLinearRegression(SKLearnMlBase):

    def __init__(self):
        self.model = linear_model.LinearRegression()
        self.model_name = "SKLearnLinearRegression"
        SKLearnMlBase.__init__(self)


class SKLearnHuberRegressor(SKLearnMlBase):

    def __init__(self):
        self.model = linear_model.HuberRegressor()
        self.model_name = "SKLearnHuberRegressor"
        SKLearnMlBase.__init__(self)






