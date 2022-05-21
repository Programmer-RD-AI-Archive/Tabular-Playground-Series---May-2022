import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import *
from torch.optim import *
from torchvision.models import *
from sklearn.model_selection import *
from sklearn.metrics import f1_score,accuracy_score,precision_score
import wandb
import nltk
from nltk.stem.porter import *
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import svm
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import warnings
import os
warnings.filterwarnings("ignore")
PROJECT_NAME = "Tabular-Playground-Series---May-2022"
np.random.seed(55)
stemmer = PorterStemmer()
device = "cuda"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")


data = pd.read_csv("./data/train.csv")
data.drop("f_27",axis=1,inplace=True)


test = pd.read_csv("./data/test.csv")
test.drop("f_27",axis=1,inplace=True)


X = data.drop("target",axis=1)
y = data['target']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.0625,shuffle=True)


from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    Normalizer,
    Binarizer
)
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from xgboost.sklearn import XGBClassifier


model = KNeighborsClassifier()
model.fit(X_train,y_train)


model.score(X_test,y_test)


ids = test['id']
preds = model.predict(test)


df = {
    "id":[],
    "target":[]
}


for _id,pred in zip(ids,preds):
    df['id'].append(_id)
    df['target'].append(pred)


df = pd.DataFrame(df)


df.to_csv('submission.csv',index=False)









