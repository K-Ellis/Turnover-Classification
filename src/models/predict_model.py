# %%writefile test.py
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from sklearn.decomposition import PCA
from pre_ml_process import pre_ml_process
from plot_confusion_matrix import plot_confusion_matrix
import pickle

# model location
model_loc = input("Please input the trained model dir:") 

# Import trained model
clf = pickle.load(open(model_loc, "rb"))

# Import Data
df_raw_loc = input("Please input the testing or prediction data dir:") 
df_raw = pd.read_csv(df_raw_loc, encoding="cp1252")

# Data cleaning
df = df_raw.dropna()
df = df.loc[df["f7"] != "#"]
df["f7"] = df["f7"].astype(float)

# f9 - remove the unknown record and binary encode the remaining two classes
df = df.loc[df["f9"] != "unknown"]
le_f9 = LabelEncoder()
df["f9"] = le_f9.fit_transform(df["f9"])

# isolate the numerical columns
numerical_cols = df.dtypes[df.dtypes != object].index.tolist()
df_num = df[numerical_cols]

# drop employee id primary key
df_num = df_num.drop("employee_id", axis=1)

# label encode string columns
def fit_label_encoders(df_in):
    fitted_label_encoders = {}
    for col in df_in.dtypes[df_in.dtypes == object].index.tolist():
        fitted_label_encoders[col] = LabelEncoder().fit(df_in[col])
    return fitted_label_encoders

fitted_label_encoders = fit_label_encoders(df.drop("employee_id", axis=1))

# concat the label encoded dataframe with the baseline dataframe 
def add_label_encoded(df_baseline, df_to_le, cols, fitted_label_encoders):
    df_out = df_baseline.copy()
    for col in cols:
        df_le = fitted_label_encoders[col].transform(df_to_le[col])
        df_out[col] = df_le
    return df_out

df_num_allLE = add_label_encoded(df_num, df, ["f1", "f2", "f3", "f4", "f10", "f12"], fitted_label_encoders)

# Separate X and y
y_col = "has_left"
y = df_num_allLE[y_col]
X = df_num_allLE.drop(y_col, axis=1)
X = X.astype(float)

# Scale predictors 
scaler = pickle.load(open("scaler.p", "rb"))
X_scaled = scaler.transform(X)

# Get predictions
y_hat = clf.predict(X_scaled)
y_proba = clf.predict_proba(X_scaled)[:,1]

# Confusion Matrix
df_cm = confusion_matrix(y, y_hat, labels=[1, 0])
plot_confusion_matrix(df_cm, 
                      target_names=[1, 0], 
                      title="%s Confusion Matrix" % (type(clf).__name__),
                      normalize=True)

# accuracy metrics
ap = average_precision_score(y, y_proba)
ps = precision_score(y, y_hat)
rs = recall_score(y, y_hat)
roc = roc_auc_score(y, y_hat)

print("average_precision_score = {:.3f}".format(ap))
print("precision_score = {:.3f}".format(ps))
print("recall_score = {:.3f}".format(rs))
print("roc_auc_score = {:.3f}".format(roc))

# Feature Importances
df_feature_importances = pd.DataFrame(clf.feature_importances_, columns=["Importance"])
col_names = df_num_allLE.columns.tolist()
col_names.remove("has_left")
df_feature_importances["Feature"] = col_names
df_feature_importances.sort_values("Importance", ascending=False, inplace=True)
df_feature_importances = df_feature_importances.round(4)
df_feature_importances = df_feature_importances.reset_index(drop=True)
print(df_feature_importances)

# concat test data with predictions
df_in_with_predictions = pd.concat([df_num_allLE, pd.Series(y_hat, name="y_hat"), pd.Series(y_proba, name="y_hat_probability")], axis=1)

# Export predictions
df_in_with_predictions.to_csv("../../data/prediction/prediction_export.csv", index=False)
