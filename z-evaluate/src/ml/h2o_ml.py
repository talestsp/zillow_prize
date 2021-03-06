import pandas as pd
import h2o
from io import StringIO
from src.utils.path_manager import PathManager
from src.dao.dao import DAO

class H2OMlBase:

    def __init__(self):
        h2o.init()
        h2o.remove_all()

        print("model built:", type(self.model).__name__)

    def train(self, df_train, target_name):
        use_cols = df_train.columns.tolist() #columns from train
        use_cols.remove(target_name) #remove target from train dataset

        parcelid_index = df_train.index.tolist()

        train_path = PathManager().get_temp_dir() + "train_temp.csv"
        df_train.to_csv(train_path, index=False)

        h2o_df_train = h2o.import_file(train_path)

        self.model.train(x=use_cols, y=target_name, training_frame=h2o_df_train)

        self.df_train = df_train
        self.target_name = target_name
        self.use_cols = use_cols

    def get_model_name(self):
        return self.model_name

    def predict(self, df_test):
        test_path = PathManager().get_temp_dir() + "test_temp.csv"
        df_test[self.use_cols].to_csv(test_path, index=False)

        h2o_df_test = h2o.import_file(test_path)
        h2o_pred = self.model.predict(h2o_df_test)

        pred = pd.read_csv(StringIO(h2o_pred.get_frame_data()), sep=",")["predict"]
        return pred.tolist()

    def r2(self):
        return self.model.r2()

    def params(self):
        return self.model.get_params()

    def columns_relevance(self):
        varimp = self.model.varimp()
        relevances = pd.DataFrame(varimp)
        relevances.columns = ["column", "1", "relevance", "3"]
        return relevances[["column", "relevance"]]

class H2OGradientBoosting(H2OMlBase):

    def __init__(self, ntrees=50):
        self.model = h2o.estimators.H2OGradientBoostingEstimator(ntrees=ntrees)
        self.model_name = "H2OGradientBoosting"
        H2OMlBase.__init__(self)


class H2ODeepLearning(H2OMlBase):

    def __init__(self, epochs=4):
        self.model = h2o.estimators.H2ODeepLearningEstimator(variable_importances=True, epochs=epochs)
        self.model_name = "H2ODeepLearning"
        H2OMlBase.__init__(self)


class H2ODeepWater(H2OMlBase):

    def __init__(self):
        self.model = h2o.estimators.H2ODeepWaterEstimator()
        self.model_name = "H2ODeepWater"
        H2OMlBase.__init__(self)






if __name__ == "__main__":
    model = H2OGradientBoosting()

    dao = DAO(train_file_name="train_complete_2016.csv")
    df_train = dao.get_normalized_data(max_na_count_columns=0.5)
    df_train = df_train.dropna()
    model.train(df_train, "logerror")

    pred = model.predict(df_train)
    print(pred)

    r2 = model.r2()
    print(r2)
