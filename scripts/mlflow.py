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
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

#get url from DVC

import dvc.api

path='train.csv'
repo='/Desktop/Pharmaceutical-Sales-prediction/scripts'
version='v1'


data_url=dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
    )

mlflow.set_experiment('demo')

def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2

if __name__=="__main__":
    warnins.filterwarnings("ignore")
    np.random.seed(40)
    
    
    #reading the csv from the remote repository
    data=pd.read_csv(data_url, sep=",")

    #log data params
    mlflow.log_param('data_url',data_url)
    mlflow.log_param('data_version',version)
    mlflow.log_param('input_rows',data.shape[0])
    mlflow.log_param('input_cols',data.shape[1])
    
    
    #split the data into training and test sets(0.75,0.25) split
    train,test=train_test_split(data)
    
    #the predicted column is 'responses' which is a scalr from[0,1]
    train_x=train.drop(["Sales"],axis=1)
    test_x=train.drop(["Sales"],axis=1)
    train_y=train[['Sales']]
    test_y=test[['Sales']]
    
    #log artifacts: columns used for modelling
    cols_x=pd.DataFrame(list(train_x.columns))
    cols_x.to_csv('features.csv',header=False,index=False)
    mlflow.log_artifact('features.csv')
    
    cols_y=pd.DataFrame(list(train_y.columns))
    cols_y.to_csv('targets.csv',header=False,index=False)
    mlflow.log_artifact('targets.csv')
    
    
    aplpha=float(sys.argv[1] if len(sys.argv)>1 else 0.5)
    li_ratio=float(sys.argv[2] if len(sys.argv)>2 else 0.5)
    
    lr=ElasticNet(aplha=alpha,li_ratio=li_ratio,random_state=42)
    lr.fit(train_x,train_y)