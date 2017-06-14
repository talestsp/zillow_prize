import pandas as pd
from src.utils.path_manager import PathManager
from src.dao.dao import DAO
from src.ml.evaluator import Evaluator
from src.ml.h2o_ml import H2OGradientBoosting, H2ODeepLearning
from src.utils.feature_selection import select_by_corr_thresh
import json

TARGET = "logerror"

def submission(model, norm, feat_selection, inputation, new_features, subm_name):
    dao = DAO(new_features=new_features)

    if norm:
        train = dao.get_normalized_data(dataset="train", inputation=inputation, max_na_count_columns=0.05)
        test = dao.get_normalized_data(dataset="test", inputation=inputation, max_na_count_columns=1)
        print(len(test))
    else:
        train = dao.get_data(cols_type="numeric", dataset="train", max_na_count_columns=0.05)
        test = dao.get_data(cols_type="numeric", dataset="test", max_na_count_columns=0.05)

    test_ids = test.index.tolist()


    if feat_selection is None:
        feat_selection_name = ""
    else:
        feat_selection_name = feat_selection.__name__
        columns = feat_selection(train)
        train_columns = columns + [TARGET]
        train = train[train_columns]
        test = test[columns]


    ev = Evaluator(model=model)
    pred = ev.run(train, test, abs_target=False)

    pred = pd.Series(pred)
    subm = pd.DataFrame()
    subm["ParcelId"] = test_ids
    subm["201610"] = pred
    subm["201611"] = pred
    subm["201612"] = pred
    subm["201710"] = pred
    subm["201711"] = pred
    subm["201712"] = pred

    subm_path = PathManager().get_submission_dir() + subm_name + ".csv"
    subm.to_csv(subm_path, index=False)

    subm_metadata = PathManager().get_submission_dir() + subm_name + ".json"
    with open(subm_metadata, 'w') as file:
        submission_dict = {}
        submission_dict["submission_name"] = subm_name
        submission_dict["norm"] = norm
        submission_dict["feat_selection"] = feat_selection_name
        submission_dict["model"] = model.get_model_name()
        submission_dict["inputation"] = inputation
        submission_dict["score"] = ""

        json.dump(submission_dict, file)



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

    submission(model=H2OGradientBoosting(), norm=False, feat_selection=None,
                      new_features=["knn-longitude-latitude"],
                      inputation="fill_0", subm_name="52f5fa93-74e7-4c74-81cc-1779ba5bfd47")



    #submission(model=H2OGradientBoosting(), norm=True, feat_selection=select_by_corr_thresh,
    #                  new_features=["knn-longitude-latitude"],
    #                  inputation="drop", subm_name="24449a9a-38e3-4115-bae8-e7a334a95c5c")



