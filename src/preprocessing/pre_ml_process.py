from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import pandas as pd

def pre_ml_process(df_in, 
                   test_size=0.33, 
                   train_test_split_random_state=42, 
                   RandomOverSampler_random_state=42):
    
    # Separate X and y
    y_col = "has_left"
    y = df_in[y_col]
    
    X = df_in.drop(y_col, axis=1)
    X = X.astype(float)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=train_test_split_random_state)

    ros = RandomOverSampler(random_state=RandomOverSampler_random_state)
    X_train_resample, y_train_resample = ros.fit_resample(X_train, y_train)
    
    # scale predictors for the linear model
    scaler = MinMaxScaler()
    X_train_resample_scaled = scaler.fit_transform(X_train_resample)
    X_test_scaled = scaler.transform(X_test)

    return df_in, X, y, X_train, X_test, y_train, y_test, \
                scaler, X_train_resample_scaled, y_train_resample, \
                X_test_scaled, ros, None

def pre_ml_process_no_resample(df_in, 
                   test_size=0.33, 
                   train_test_split_random_state=42, 
                   RandomOverSampler_random_state=42):
    
    # Separate X and y
    y_col = "has_left"
    y = df_in[y_col]
    
    X = df_in.drop(y_col, axis=1)
    X = X.astype(float)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=train_test_split_random_state)

    ros = None
    X_train_resample, y_train_resample = X_train, y_train
    
    # scale predictors for the linear model
    scaler = MinMaxScaler()
    X_train_resample_scaled = scaler.fit_transform(X_train_resample)
    X_test_scaled = scaler.transform(X_test)
    
    return df_in, X, y, X_train, X_test, y_train, y_test, \
                scaler, X_train_resample_scaled, y_train_resample, \
                X_test_scaled, ros, None


def pre_ml_process_resample_PolyFeatures(df_in, 
                   test_size=0.33, 
                   train_test_split_random_state=42, 
                   RandomOverSampler_random_state=42):
    
    # Separate X and y
    y_col = "has_left"
    y = df_in[y_col]
    
    X = df_in.drop(y_col, axis=1)
    X = X.astype(float)
    
    # Make a new dataframe for polynomial features
    input_poly_features_list = ['f5', 'f6', 'f7', 'f8', 'f11', 'f13']
    poly_features = X.loc[:,X.columns.isin(input_poly_features_list)]

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree = 3)
    
    # Fit and transform the poly_transformer to the data
    poly_features = poly_transformer.fit_transform(poly_features)
    
    poly_features = pd.DataFrame(
        poly_features, 
        columns = poly_transformer.get_feature_names(input_features = input_poly_features_list)
    )
    poly_features = poly_features.drop(input_poly_features_list, axis=1)
    if "1" in poly_features.columns:
        poly_features = poly_features.drop("1", axis=1)
    
    X = X.join(poly_features)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=train_test_split_random_state)

    ros = RandomOverSampler(random_state=RandomOverSampler_random_state)
    X_train_resample, y_train_resample = ros.fit_resample(X_train, y_train)
    
    # scale predictors for the linear model
    scaler = MinMaxScaler()
    X_train_resample_scaled = scaler.fit_transform(X_train_resample)
    X_test_scaled = scaler.transform(X_test)
    
    df_out = pd.concat([X, y], axis=1)

    return df_out, X, y, X_train, X_test, y_train, y_test, \
                scaler, X_train_resample_scaled, y_train_resample, \
                X_test_scaled, ros, poly_transformer
