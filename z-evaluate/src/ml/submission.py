import pandas as pd
from src.utils.path_manager import PathManager
from src.utils.process_data import process_data
from src.ml.evaluator import Evaluator
from src.ml.h2o_ml import H2ODeepLearning, H2OGradientBoosting
from src.ml.sklearn_ml import SKLearnLinearRegression, SKLearnHuberRegressor
from src.dao.dao import DAO
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


    train = process_data(dataset="train", cols_type=cols_type, norm=norm, inputation=inputation,
                        new_features=new_features, feat_selection=feat_selection)

    test = process_data(dataset="test", cols_type=cols_type, norm=norm, inputation=inputation,
                         new_features=new_features, feat_selection=None)

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

    test_ids = test.index.tolist()

    make_submission_file(id, pred, test_ids, data_eval)

# def process_data(dataset, cols_type, norm, inputation, new_features, feat_selection, max_na_count_columns=0.05):
#     if dataset == "test":
#         max_na_count_columns = 1.0
#
#     dao = DAO(new_features=new_features)
#
#     if norm and cols_type == "numeric":
#         df = dao.get_normalized_data(dataset=dataset, inputation=inputation, max_na_count_columns=max_na_count_columns)
#
#     elif norm and cols_type == "all":
#         df_norm = dao.get_normalized_data(dataset=dataset, inputation=inputation, max_na_count_columns=max_na_count_columns)
#
#         if "parcelid" in df_norm.columns.tolist():
#             del df_norm["parcelid"]
#
#         df = dao.get_data(dataset=dataset, inputation=inputation, cols_type=cols_type, max_na_count_columns=max_na_count_columns)
#
#         if "parcelid" in df.columns.tolist():
#             del df["parcelid"]
#
#         for numeric_col in df_norm.columns.tolist():
#             if numeric_col != TARGET:
#                 del df[numeric_col]
#
#         if dataset == "train":
#             on_cols = ['parcelid', TARGET]
#         else:
#             on_cols = ['parcelid']
#
#         df = pd.merge(df.reset_index(), df_norm.reset_index(), on=on_cols, how="left").set_index('parcelid')
#
#         df_norm = None
#         gc.collect()
#
#     else:
#         df = dao.get_data(cols_type=cols_type, dataset=dataset, inputation=inputation, max_na_count_columns=max_na_count_columns)
#
#     if dataset == "train" and not feat_selection is None:
#         if feat_selection == "select_by_corr_thresh":
#             feat_selection = select_by_corr_thresh
#
#         columns = feat_selection(df) + [TARGET]
#         df = df[columns]
#
#     return df

def pick_model(model_name):
    if model_name == "H2OGradientBoosting":
        return H2OGradientBoosting()

    if model_name == "H2ODeepLearning":
        return H2ODeepLearning()

    if model_name == "SKLearnLinearRegression":
        return SKLearnLinearRegression()

    if model_name == "SKLearnHuberRegressor":
        return SKLearnHuberRegressor()

def make_submission_file(id, pred, test_ids, data_eval):
    print("\n\n")
    print("id:", id)
    subm_name = id
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


    submission_by_id(id="1e3228ec-3876-4b7f-b3bf-0e7717266529")
