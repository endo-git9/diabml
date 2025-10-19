# core/data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RND = 42

def load_dataset(path):
    df = pd.read_csv(path)
    if "Outcome" not in df.columns:
        raise ValueError("Dataset must contain an 'Outcome' column.")
    return df

def train_test_calib_split(df, test_size=0.3, calib_size=0.2, random_state=RND):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, random_state=random_state)
    # create calibration split from train
    X_train_sub, X_calib, y_train_sub, y_calib = train_test_split(
        X_train, y_train, test_size=calib_size, stratify=y_train, random_state=random_state
    )
    return X_train_sub, X_calib, X_test, y_train_sub, y_calib, y_test

def resample_and_scale(X_train, y_train, X_test, random_state=RND):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, y_res, X_test_scaled, scaler