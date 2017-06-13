import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.dao.dao import DAO
from src.ml.h2o_ml import H2ODeepLearning, H2OGradientBoosting
from src.ml.sklearn_ml import SKLearnLinearRegression, SKLearnRANSACRegression
from src.utils.partitioner import simple_partition
from src.utils.results import Results
from src.utils.feature_selection import select_by_corr_thresh

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

TARGET = "logerror"


class Evaluator:

    def __init__(self, df, model):
        self.df = df
        self.model = model

    def evaluate(self, train_part_size=0.7, abs_target=False):
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

        self.results = self.build_results(mae, self.model, result_df)

        return self.results

    def run(self, train_part, test_part):
        self.model.train(train_part, target_name=TARGET)

        if TARGET in test_part.columns.tolist():
            del test_part[TARGET]

        predict = self.model.predict(test_part)

        return predict

    def build_results(self, mae, model, result_df):

        results = Results(model=model, result_df=result_df, mae=mae)
        return results

    def get_results(self):
        return self.results


if __name__ == "__main__":

    for feat_selection in [select_by_corr_thresh, None]:
        for model in [SKLearnRANSACRegression(), SKLearnLinearRegression()]:
            for new_features in [["knn-longitude-latitude"], []]:
                for abs_target in [True, False]:
                    for norm in [True, False]:
                        print("Evaluating:", model.__class__.__name__)
                        tags = []
                        dao = DAO(df_file_name="train_complete_2016.csv", new_features=new_features)
                        if norm:
                            df = dao.get_normalized_data(max_na_count_columns=0.05)
                            tags.append("norm")

                        else:
                            df = dao.get_data(cols_type="numeric", max_na_count_columns=0.05)

                        if abs_target:
                            tags.append("abs")

                        df = df.dropna()

                        if not feat_selection is None:
                            columns = feat_selection(df) + [TARGET]
                            df = df[columns]

                        ev = Evaluator(df, model=model)
                        ev.evaluate(train_part_size=0.7, abs_target=abs_target)

                        ev.get_results().set_tags(tags=tags)
                        ev.get_results().set_new_features(new_features=new_features)
                        ev.get_results().set_feat_selection(feat_selection=str(feat_selection))
                        ev.get_results().print()
                        ev.get_results().save()

