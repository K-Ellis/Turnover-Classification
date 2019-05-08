# %%writefile train.py
# Import Packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from pre_ml_process import pre_ml_process
import pickle
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
%matplotlib inline

# location of training data
training_data_loc = input("Please input the training data dir:") 

# Import Data
df_raw = pd.read_csv(training_data_loc, encoding="cp1252")

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


XGC=xgb.XGBClassifier(random_state=0, n_estimators=100) 

# parameters
split_random_state=42
xgb_fit_eval_metric="aucpr"
train_test_split_random_state=0
RandomOverSampler_random_state=0
test_size=0.33

# preprocessing
df_ignore, X, y, X_train, X_test, y_train, y_test, \
                scaler, X_train_resample_scaled, y_train_resample, \
                X_test_scaled, ros, poly_ignore = \
                pre_ml_process(df_num_allLE, 
                               test_size,
                               train_test_split_random_state,
                               RandomOverSampler_random_state)

# save scaler to file
pickle.dump(scaler, open("../../models/scaler.p", "wb"))

# Train with XGBoost Classifier
clf_XG = XGC.fit(X_train_resample_scaled, y_train_resample, eval_metric=xgb_fit_eval_metric)

# Model evaluation

# Get test set predictions
y_test_hat = clf_XG.predict(X_test_scaled)
y_test_proba = clf_XG.predict_proba(X_test_scaled)[:,1]

# Confusion Matrix
df_cm = confusion_matrix(y_test, y_test_hat, labels=[1, 0])
plot_confusion_matrix(df_cm, 
                      target_names=[1, 0], 
                      title="%s Confusion Matrix" % (type(clf_XG).__name__),
                      normalize=True)
plt.show()

# Accuracy metrics
ap = average_precision_score(y_test, y_test_proba)
ps = precision_score(y_test, y_test_hat)
rs = recall_score(y_test, y_test_hat)
roc = roc_auc_score(y_test, y_test_hat)

print("average_precision_score = {:.3f}".format(ap))
print("precision_score = {:.3f}".format(ps))
print("recall_score = {:.3f}".format(rs))
print("roc_auc_score = {:.3f}".format(roc))

# Feature Importances
df_feature_importances = pd.DataFrame(clf_XG.feature_importances_, columns=["Importance"])
col_names = df_num_allLE.columns.tolist()
col_names.remove("has_left")
df_feature_importances["Feature"] = col_names
df_feature_importances.sort_values("Importance", ascending=False, inplace=True)
df_feature_importances = df_feature_importances.round(4)
df_feature_importances = df_feature_importances.reset_index(drop=True)
print(df_feature_importances)

# export trained model
pickle.dump(clf_XG, open("../../models/xgb_model.p", "wb"))