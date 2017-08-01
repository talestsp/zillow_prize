import pandas as pd
from src.utils.path_manager import PathManager
from src.utils.process_data import process_data
from src.ml.evaluator import Evaluator
from src.ml.h2o_ml import H2ODeepLearning, H2OGradientBoosting
from src.ml.sklearn_ml import SKLearnLinearRegression, SKLearnHuberRegressor
from src.dao.dao import DAO
from src.utils.feature_selection import select_by_corr_thresh
import json
import gc

TARGET = "logerror"

def get_data_eval(id):
    filepath = PathManager().get_results_data_eval_dir() + id + ".json"

    with open(filepath, 'r') as file:
        json_data_eval = json.load(file)
        # print(json_data_eval)

        return json_data_eval

def submission_by_id(id):
    data_eval = get_data_eval(id)
    print(data_eval)

    abs_target = data_eval["abs"]
    cols_type = data_eval["cols_type"]
    feat_selection = data_eval["feat_selection"]
    new_features = data_eval["new_features"]
    norm = data_eval["norm"]
    inputation = data_eval["inputation"]
    model_name = data_eval["model_name"]


    if feat_selection == "select_by_corr_thresh":
        feat_selection = select_by_corr_thresh


    dao = DAO(new_features=new_features)

    train = process_data(dao=dao, dataset="train", cols_type=cols_type, norm=norm, inputation=inputation,
                        new_features=new_features, feat_selection=feat_selection, max_na_count_columns=1.0)

    test = process_data(dao=dao, dataset="test", cols_type=cols_type, norm=norm, inputation=inputation,
                         new_features=new_features, feat_selection=None, max_na_count_columns=1.0)

    use_cols = train.columns.tolist()
    use_cols.remove(TARGET)
    test = test[use_cols]

    dao = None
    gc.collect()

    model = pick_model(model_name)
    ev = Evaluator(model=model)
    print("READY!!!!")
    print("train", train.shape)
    print(train.head())
    print()
    print("test", test.shape)
    print(test.head())

    pred = ev.run(train, test, abs_target=abs_target)
    pred = pd.Series(pred)

    print("Predictions length:", len(pred))
    print(pred.head())

    test_ids = test.index.tolist()

    make_submission_file(id, pred, test_ids, data_eval)

def pick_model(model_name):
    if model_name == "H2OGradientBoosting":
        return H2OGradientBoosting(ntrees=100)

    if model_name == "H2ODeepLearning":
        return H2ODeepLearning(epochs=10)

    if model_name == "SKLearnLinearRegression":
        return SKLearnLinearRegression()

    if model_name == "SKLearnHuberRegressor":
        return SKLearnHuberRegressor()

def make_submission_file(id, pred, test_ids, data_eval):
    print("\n\n")
    print("id:", id)
    subm_name = id
    pred = pd.Series(pred).round(8)
    subm = pd.DataFrame()
    subm["ParcelId"] = test_ids
    subm["201610"] = pred
    subm["201611"] = pred
    subm["201612"] = pred
    subm["201710"] = pred
    subm["201711"] = pred
    subm["201712"] = pred

    print("submission")
    print(subm)
    subm_path = PathManager().get_submission_dir() + subm_name + ".csv"
    subm.to_csv(subm_path, index=False)

    subm_metadata = PathManager().get_submission_dir() + subm_name + ".json"
    with open(subm_metadata, 'w') as file:
        submission_data = {}
        submission_data["id"] = id
        submission_data["score"] = ""

        json.dump(submission_data, file)



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

    pred = pd.Series(pred).round(10)
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

    # submission(model=H2OGradientBoosting(), norm=False, feat_selection=None,
    #                   new_features=["knn-longitude-latitude"],
    #                   inputation="fill_0", subm_name="52f5fa93-74e7-4c74-81cc-1779ba5bfd47")



    #submission(model=H2OGradientBoosting(), norm=True, feat_selection=select_by_corr_thresh,
    #                  new_features=["knn-longitude-latitude"],
    #                  inputation="drop", subm_name="24449a9a-38e3-4115-bae8-e7a334a95c5c")


    submission_by_id(id="0b0fc084-ebef-4bcd-b896-c06a38b04c53")
