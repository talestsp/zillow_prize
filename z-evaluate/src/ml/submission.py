import pandas as pd
from src.utils.path_manager import PathManager
from src.dao.dao import DAO
from src.ml.evaluator import Evaluator
from src.ml.h2o_ml import H2OGradientBoosting, H2ODeepLearning
from src.utils.feature_selection import select_by_corr_thresh

def submission(pred, index):
    pred = pd.Series(pred)
    pred = round(pred, 5)
    subm = pd.DataFrame()
    subm["ParcelId"] = index
    subm["201610"] = pred
    subm["201611"] = pred
    subm["201612"] = pred
    subm["201710"] = pred
    subm["201711"] = pred
    subm["201712"] = pred
    return subm



if __name__ == "__main__":

    dao = DAO(new_features=["knn-longitude-latitude"])

    train = dao.get_normalized_data(dataset="train", max_na_count_columns=0.05)
    test = dao.get_normalized_data(dataset="test", max_na_count_columns=0.05)

    teste_not_norm = dao.get_data(cols_type="numeric", dataset="train", max_na_count_columns=0.05)

    test_ids = test.index.tolist()

    columns = select_by_corr_thresh(train)
    train_columns = columns + ["logerror"]

    ev = Evaluator(model=H2ODeepLearning())
    pred = ev.run(train[train_columns], test[columns], abs_target=False)

    subm = submission(pred, test_ids)
    print(subm.head())

    subm_path = PathManager().get_submission_dir() + "sub2.csv"
    subm.to_csv(subm_path, index=False)



