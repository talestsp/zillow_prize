import os

PROJECT_DIR = "z-evaluate/"
RESULTS_DIR = "results/"
PLOTS_DIR = "plots/"
PREDICTIONS_EVAL_DIR = "predictions_eval/"
DATA_EVAL_DIR = "data_eval/"
DATA_DIR = "data/"
NEW_FEATURES_DIR = "new_features/"
SUBMISSION_DIR = "submissions/"
TEMP_DIR = "temp/"

class PathManager:
    def __init__(self):
        self.root_dir = self.root_directory()

    def root_directory(self):
        wd_split = os.path.dirname(os.path.realpath(__file__)).split("/")
        i = wd_split.index("z-evaluate")
        return "/".join(wd_split[0:i]) + "/"

    def get_results_dir(self):
        path = self.root_dir + RESULTS_DIR
        self.create_dir(path)
        return path

    def get_results_plot_dir(self):
        path = self.root_dir + RESULTS_DIR + PLOTS_DIR
        self.create_dir(path)
        return path

    def get_results_predictions_eval_dir(self):
        path = self.root_dir + RESULTS_DIR + PREDICTIONS_EVAL_DIR
        self.create_dir(path)
        return path

    def get_results_data_eval_dir(self):
        path = self.root_dir + RESULTS_DIR + DATA_EVAL_DIR
        self.create_dir(path)
        return path

    def get_new_features_dir(self):
        return self.root_dir + DATA_DIR + NEW_FEATURES_DIR

    def get_data_dir(self, data_file_name):
        return self.root_dir + PROJECT_DIR + DATA_DIR + data_file_name

    def get_submission_dir(self):
        path = self.root_dir + SUBMISSION_DIR
        self.create_dir(path)
        return path

    def get_temp_dir(self):
        path = self.root_dir + TEMP_DIR
        self.create_dir(path)
        return path

    def create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)



if __name__ == "__main__":
    PathManager()
