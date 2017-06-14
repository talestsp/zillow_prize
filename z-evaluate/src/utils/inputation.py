from sklearn.preprocessing import Imputer
from src.dao.dao import DAO

def inputtation(df, strategy="most_frequent"):
    inputer = Imputer(strategy=strategy, axis=0)



dao = DAO(train_file_name="train_complete_2016.csv", new_features=["knn-longitude-latitude"])

df = dao.get_data(cols_type="numeric", max_na_count_columns=0.05)

inputer = Imputer(strategy="most_frequent", axis=0)

