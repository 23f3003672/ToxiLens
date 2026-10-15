from typing import Tuple 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 150, random_state: int=42,) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight ="balanced", random_state=random_state,n_jobs=-1)
    clf.fit(X_train,y_train)
    return clf 

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 150,max_depth: int=5, learning_rate: float=0.05, random_state: int=42,) -> XGBClassifier:

    num_neg = int(np.sum(y_train == 0))
    num_pos = int(np.sum(y_train == 1))
    scale_pos = (num_neg/num_pos) if num_pos > 0 else 1.0

    clf = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,scale_pos_weight=scale_pos,eval_metric = "logloss", random_state=random_state, n_jobs=-1)
    clf.fit(X_train,y_train)
    return clf 

def get_predictions(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:,1] 
