import pandas as pd
from src.utils.path_manager import PathManager
from src.dao.dao import DAO
from src.ml.evaluator import Evaluator
from src.ml.h2o_ml import H2OGradientBoosting, H2ODeepLearning
from src.utils.feature_selection import select_by_corr_thresh
import json

TARGET = "logerror"

def submission(model, norm, feat_selection, new_features, subm_name):
    dao = DAO(new_features=new_features)

    if feat_selection is None:
        feat_selection_name = ""
    else:
        feat_selection_name = feat_selection.__name__

    if norm:
        train = dao.get_normalized_data(dataset="train", max_na_count_columns=0.05)
        test = dao.get_normalized_data(dataset="test", max_na_count_columns=1)
        print(len(test))
    else:
        train = dao.get_data(cols_type="numeric", dataset="train", max_na_count_columns=0.05)
        test = dao.get_data(cols_type="numeric", dataset="test", max_na_count_columns=0.05)

    # test_ids = test.index.tolist()
    #
    # columns = feat_selection(train)
    # train_columns = columns + [TARGET]
    #
    # ev = Evaluator(model=model)
    # pred = ev.run(train[train_columns], test[columns], abs_target=False)
    #
    # pred = pd.Series(pred)
    # pred = round(pred, 5)
    # subm = pd.DataFrame()
    # subm["ParcelId"] = test_ids
    # subm["201610"] = pred
    # subm["201611"] = pred
    # subm["201612"] = pred
    # subm["201710"] = pred
    # subm["201711"] = pred
    # subm["201712"] = pred
    #
    # subm_path = PathManager().get_submission_dir() + subm_name + ".csv"
    # subm.to_csv(subm_path, index=False)
    #
    # subm_metadata = PathManager().get_submission_dir() + subm_name + ".json"
    # with open(subm_metadata, 'w') as file:
    #     submission_dict = {}
    #     submission_dict["submission_name"] = subm_name
    #     submission_dict["norm"] = norm
    #     submission_dict["feat_selection"] = feat_selection_name
    #     submission_dict["model"] = model.get_model_name()
    #
    #     json.dump(submission_dict, file)



if __name__ == "__main__":

    # dao = DAO(new_features=["knn-longitude-latitude"])
    #
    # train = dao.get_normalized_data(dataset="train", max_na_count_columns=0.05)
    # test = dao.get_normalized_data(dataset="test", max_na_count_columns=0.05)
    #
    # test_ids = test.index.tolist()
    #
    # columns = select_by_corr_thresh(train)
    # train_columns = columns + ["logerror"]
    #
    # ev = Evaluator(model=H2ODeepLearning())
    # pred = ev.run(train[train_columns], test[columns], abs_target=False)
    #
    # subm = submission(pred, test_ids)
    # print(subm.head

    subm = submission(model=H2ODeepLearning(), norm=True, feat_selection=select_by_corr_thresh,
                      new_features=["knn-longitude-latitude"], subm_name="sub2.csv")

    # subm_path = PathManager().get_submission_dir() + "sub2.csv"
    # subm.to_csv(subm_path, index=False)



