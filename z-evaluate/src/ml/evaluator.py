import pandas as pd
from sklearn.metrics import mean_absolute_error
import h2o

from src.dao.dao import DAO
from src.ml.partitioner import simple_partition
from src.ml.sklearn_ml import SKLearnLinearRegression, SKLearnLasso
from src.ml.h2o_ml import H2OGradientBoosting, H2ODeepLearning, H2ODeepWater, H2OStackedEnsemble
from src.utils.results import Results

pd.set_option('display.float_format', lambda x: '%.5f' % x)

TARGET = "logerror"


class Evaluator:

    def __init__(self, df, model):
        self.df = df
        self.model = model

    def evaluate(self, train_part_size=0.7, abs_target=False, tags=[]):
        '''

        :param train_part_size: proportion of dataset to be set as train partition
        :param abs_target: converts target values to positives values by aplying abs()
        :param tags: experiment tags, a list of strings
        :return: Results object
        '''
        use_df = df.copy()
        if abs_target:
            use_df[TARGET] = abs(use_df[TARGET])

        train_part, test_part = simple_partition(use_df, train_part_size)
        test_part_target = test_part[TARGET]

        predict = self.run(train_part, test_part)

        result_df = pd.DataFrame({"real": test_part_target, "prediction": predict})
        mae = mean_absolute_error(result_df["real"], result_df["prediction"])

        self.results = self.build_results(mae, self.model, result_df, tags)

        return self.results

    def run(self, train_part, test_part):
        self.model.train(train_part, target_name=TARGET)

        if TARGET in test_part.columns.tolist():
            del test_part[TARGET]

        predict = self.model.predict(test_part)

        return predict

    def build_results(self, mae, model, result_df, tags):
        r2 = model.r2()

        results = Results(model=model, result_df=result_df,
                          mae=mae, r2=r2, tags=tags)

        return results

    def get_results(self):
        return self.results


if __name__ == "__main__":
    h2o.init()
    h2o.remove_all()

    for i in [0, 1]:
        for new_features in [["knn-longitude-latitude"], []]:

            for model in [H2ODeepLearning(), H2ODeepWater(), SKLearnLinearRegression(), H2OGradientBoosting(),
                          H2OStackedEnsemble(),  SKLearnLasso()]:
                for abs_target in [True, False]:
                    for norm in [True, False]:
                        tags = []

                        if abs_target:
                            tags.append("abs")

                        if norm:
                            dao = DAO(df_file_name="train_complete_2016.csv", new_features=new_features)
                            df = dao.get_normalized_data(max_na_count_columns=0.05)
                            tags.append("norm")
                        else:
                            df = dao.get_data(max_na_count_columns=0.05)

                        df = df.dropna()

                        try:
                            ev = Evaluator(df, model=model)
                            ev.evaluate(train_part_size=0.7, tags=tags, abs_target=abs_target)

                        except:
                            df = dao.get_data(cols_type="numeric", max_na_count_columns=0.05)
                            df = df.dropna()
                            tags.append("numeric")
                            ev = Evaluator(df, model=model)
                            ev.evaluate(train_part_size=0.7, tags=tags, abs_target=abs_target)


                        try:
                            ev.get_results().print()
                        except:
                            pass

                        ev.get_results().set_new_features(new_features)
                        ev.get_results().save()

