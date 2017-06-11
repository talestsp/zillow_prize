from uuid import uuid4
from datetime import datetime
import pytz
import json

from src.plot.plot import prediction_scatter_plot
from src.utils.path_manager import PathManager


class Results:

    def __init__(self, model_name, result_df,  params, mae, r2, tags=[]):
        self.id = uuid4()
        self.datetime = datetime.now(pytz.timezone("Brazil/East")).__str__().split(".")[0]
        self.model_name = model_name
        self.result_df = result_df
        self.params = params
        self.mae = mae
        self.r2 = r2
        self.tags = tags

    def add_tags(self, tags):
        self.tags.append(tags)

    def save(self):
        results_file_path = PathManager().get_results_data_eval_dir() + self.id.__str__() + ".json"
        with open(results_file_path, 'w') as file:
            json.dump(self.result_dict(), file)

        plot_file_path = PathManager().get_results_plot_dir() + self.id.__str__() + ".html"
        self.plot(show=False, save=True, file_name=plot_file_path)

        result_df_file_path = PathManager().get_results_predictions_eval_dir() + self.id.__str__() + ".csv"
        self.result_df.to_csv(result_df_file_path, index=False)

    def show_plot(self):
        self.plot(show=True, save=False)

    def plot(self, show, save, file_name=None):
        title = self.model_name + " " + "mae:" + str(round(self.mae, 5)) + " " + "r2:" + str(round(self.r2, 5)) + " " + "tags:" + str(self.tags)
        prediction_scatter_plot(self.result_df, show_plot=show, save_plot=save, title=title, file_name=file_name)

    def print(self):
        print("id:", self.id)
        print("Date:", self.datetime)
        print("MAE:", self.mae)
        print("R2:", self.r2)
        print("tags:", self.tags)
        print("params:", self.params)
        print()

    def result_dict(self):
        result_dict = {}
        result_dict["id"] = self.id.__str__()
        result_dict["date"] = self.datetime
        result_dict["model_name"] = self.model_name
        result_dict["mae"] = self.mae
        result_dict["r2"] = self.r2
        result_dict["tags"] = self.tags
        result_dict["params"] = self.params

        return result_dict

    def result_json(self):
        result_dict = self.result_dict()
        return json.dumps(result_dict)
