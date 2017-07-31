import pandas as pd
from sklearn.metrics import mean_absolute_error
import gc

from src.ml.h2o_ml import H2ODeepLearning, H2OGradientBoosting
from src.ml.sklearn_ml import SKLearnLinearRegression, SKLearnHuberRegressor
from src.utils.partitioner import simple_partition
from src.utils.results import Results
from src.utils.feature_selection import select_by_corr_thresh
from src.utils.process_data import process_data

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

TARGET = "logerror"


class Evaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, data, train_part_size=0.7, abs_target=False):
        '''
        :param train_part_size: proportion of dataset to be set as train partition
        :param abs_target: converts target values to positives values by aplying abs()
        :return: Results object
        '''
        use_df = data.copy()

        train_part, test_part = simple_partition(use_df, train_part_size)
        test_part_target = test_part[TARGET]

        predict = self.run(train_part, test_part, abs_target)

        result_df = pd.DataFrame({"real": test_part_target, "prediction": predict})
        mae = mean_absolute_error(result_df["real"], result_df["prediction"])

        self.results = self.build_results(mae, self.model, result_df)

        return self.results

    def run(self, train_part, test_part, abs_target):
        if abs_target:
            train_part[TARGET] = abs(train_part[TARGET])
            test_part[TARGET] = abs(test_part[TARGET])

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

    cont = 0
    for abs_target in [False, True]:
        for cols_type in ["all", "numeric"]:
            for feat_selection in [select_by_corr_thresh, None]:
                for new_features in [[], ["knn-longitude-latitude"], ["knn-longitude-latitude-signal"], ["knn-longitude-latitude", "knn-longitude-latitude-signal"]]:
                    for norm in [True, False]:
                        for inputation in ["column_mean", "fill_0"]:
                            for model in [H2OGradientBoosting(), H2ODeepLearning(), SKLearnLinearRegression(), SKLearnHuberRegressor()]:
                                print("\n\n\n")
                                cont += 1
                                print("essay:", cont)

                                df = process_data(dataset="train", cols_type=cols_type, norm=norm,
                                                  inputation=inputation,
                                                  new_features=new_features, feat_selection=feat_selection,
                                                  max_na_count_columns=0.05)


                                # dao = DAO(train_file_name="train_complete_2016.csv", new_features=new_features)
                                #
                                # if norm and cols_type == "numeric":
                                #     df = dao.get_normalized_data(dataset="train", inputation=inputation, max_na_count_columns=0.05)
                                #
                                # elif norm and cols_type == "all":
                                #     df_norm = dao.get_normalized_data(dataset="train", inputation=inputation, max_na_count_columns=0.05)
                                #
                                #     if "parcelid" in df_norm.columns.tolist():
                                #         del df_norm["parcelid"]
                                #
                                #     df = dao.get_data(cols_type=cols_type, inputation=inputation, max_na_count_columns=0.05)
                                #
                                #     if "parcelid" in df.columns.tolist():
                                #         del df["parcelid"]
                                #
                                #     for numeric_col in df_norm.columns.tolist():
                                #         if numeric_col != TARGET:
                                #             del df[numeric_col]
                                #
                                #     df = pd.merge(df.reset_index(), df_norm.reset_index(), on=['parcelid', TARGET]).set_index('parcelid')
                                #
                                #     df_norm = None
                                #     gc.collect()
                                #
                                # else:
                                #     df = dao.get_data(dataset="train", inputation=inputation, cols_type=cols_type, max_na_count_columns=0.05)
                                #
                                # df = df.dropna()
                                #
                                # if not feat_selection is None:
                                #     columns = feat_selection(df) + [TARGET]
                                #     df = df[columns]
                                #     feature_selection_name = feat_selection.__name__
                                #
                                # else:
                                #     feature_selection_name = ""

                                print("Evaluating:", model.__class__.__name__)
                                ev = Evaluator(model=model)

                                try:
                                    ev.evaluate(df, train_part_size=0.7, abs_target=abs_target)

                                    ev.get_results().set_cols_type(cols_type=cols_type)
                                    ev.get_results().set_norm(norm=norm)

                                    ev.get_results().set_new_features(new_features=new_features)
                                    if feat_selection is None:
                                        ev.get_results().set_feat_selection(feat_selection="")
                                    else:
                                        ev.get_results().set_feat_selection(feat_selection=feat_selection.__name__)
                                    ev.get_results().set_inputation(inputation=inputation)
                                    ev.get_results().set_abs_target(abs_target=abs_target)
                                    ev.get_results().save()
                                    ev.get_results().print()

                                except ValueError as e:
                                    print(e)


