import os

PROJECT_DIR = "z-evaluate/"
RESULTS_DIR = "results/"
PLOTS_DIR = "plots/"
PREDICTIONS_EVAL_DIR = "predictions_eval/"
DATA_EVAL_DIR = "predictions_eval/"
DATA_DIR = "data/"

class PathManager:
    def __init__(self):
        self.root_dir = self.root_directory()

    def root_directory(self):
        wd_split = os.path.dirname(os.path.realpath(__file__)).split("/")
        i = wd_split.index("z-evaluate")
        return "/".join(wd_split[0:i]) + "/"

    def get_results_dir(self):
        path = self.root_dir + PROJECT_DIR + RESULTS_DIR
        self.create_dir(path)
        return path

    def get_results_plot_dir(self):
        path = self.get_results_dir() + PLOTS_DIR
        self.create_dir(path)
        return path

    def get_results_predictions_eval_dir(self):
        path = self.get_results_dir() + PREDICTIONS_EVAL_DIR
        self.create_dir(path)
        return path

    def get_results_data_eval_dir(self):
        path = self.get_results_dir() + DATA_EVAL_DIR
        self.create_dir(path)
        return path

    def get_data_dir(self, data_file_name):
        return self.root_dir + DATA_DIR + data_file_name

    def create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)



if __name__ == "__main__":
    PathManager()
