import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__=="__main__":
    warnins.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_path = r"C:\Users\user.DESKTOP-OMQ89VA\Desktop\Pharmaceutical-Sales-prediction\data\train.csv"
    data = pd.read_csv(csv_path, sep=";")
    

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["Sales"], axis=1)
    test_x = test.drop(["Sales"], axis=1)
    train_y = train[["Sales"]]
    test_y = test[["Sales"]]

    #alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    #l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    