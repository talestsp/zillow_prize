import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.dao.dao import DAO
from src.ml.partitioner import simple_partition
from src.ml.linear_regression import LinearRegression, RANSACRegression
from src.utils.results import Results

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

        if abs_target:
            self.results.add_tags("abs")
        return self.results

    def run(self, train_part, test_part):
        self.model.train(train_part, target_name=TARGET)

        if TARGET in test_part.columns.tolist():
            del test_part[TARGET]

        predict = self.model.predict(test_part)

        return predict

    def build_results(self, mae, model, result_df, tags):
        r2 = model.r2()

        results = Results(model_name=model.get_model_name(), result_df=result_df,  params=model.params(),
                          mae=mae, r2=r2, tags=tags)
        results.result_json()

        return results

    def get_results(self):
        return self.results


if __name__ == "__main__":

    for model in [LinearRegression(), RANSACRegression()]:
        for abs_target in [False, True]:
            for norm in [False,  True]:
                dao = DAO(df_file_name="train_complete_2016.csv")
                if norm:
                    df = dao.get_normalized_data(max_na_count_columns=0.05)
                else:
                    df = dao.get_data(cols_type="numeric", max_na_count_columns=0.05)

                df = df.dropna()
                ev = Evaluator(df, model=model)
                ev.evaluate(train_part_size=0.7, tags=[], abs_target=abs_target)
                ev.get_results().print()
                ev.get_results().save()

